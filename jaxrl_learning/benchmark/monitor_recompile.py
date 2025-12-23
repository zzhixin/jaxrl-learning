"""
Donation-safe JAX jit recompile monitor.

Features:
- Best-effort inference of static_argnames from a jitted function
- On each NEW compilation (cache_size increases):
    * If static signature is new: report the minimal static args that "explain" the new variant,
      and print all previously cached values for those modified static args.
    * If static signature was already seen: investigate deeper by diffing a more faithful
      dynamic signature (shape/dtype/weak_type + device/sharding) against the closest cached
      variant with the same static signature; also fingerprints non-array args safely.
- Donation-safe: NEVER repr() donated arrays or containers that contain arrays.

Usage:
  f_mon = monitor_recompiles_explain_deeper(f)   # wraps jitted f
  f_mon(*args, **kwargs)                         # call like normal
"""

import inspect
import functools
import jax
import jax.numpy as jnp


# ----------------------------
# Static argname inference
# ----------------------------
def infer_static_argnames(fn):
    """Best-effort inference of static_argnames from a jitted function."""
    if isinstance(fn, functools.partial):
        san = fn.keywords.get("static_argnames", None) if fn.keywords else None
        if san is not None:
            return tuple(san)

    for attr in ("static_argnames", "_static_argnames", "__static_argnames__"):
        san = getattr(fn, attr, None)
        if san:
            try:
                return tuple(san)
            except TypeError:
                return (san,)

    for attr in ("jit_info", "_jit_info", "_cache_miss", "_pjit_info", "_info"):
        info = getattr(fn, attr, None)
        if info is None:
            continue
        for key in ("static_argnames", "_static_argnames"):
            san = getattr(info, key, None)
            if san:
                try:
                    return tuple(san)
                except TypeError:
                    return (san,)

    wrapped = getattr(fn, "__wrapped__", None)
    if wrapped is not None and wrapped is not fn:
        return infer_static_argnames(wrapped)

    return ()


# ----------------------------
# Donation-safe fingerprinting
# ----------------------------
def _safe_repr(obj, maxlen=200):
    """Best-effort repr that won’t crash on donated arrays."""
    try:
        s = repr(obj)
        if len(s) > maxlen:
            s = s[:maxlen] + "..."
        return s
    except Exception as e:
        return f"<repr failed: {type(e).__name__}: {e}>"


def _is_jax_array(obj) -> bool:
    # Covers jax.Array and jaxlib internal array impl types
    return isinstance(obj, jax.Array) or type(obj).__module__.startswith("jaxlib")


def _safe_array_fingerprint(a):
    """Fingerprint a JAX array without touching its value (safe under donation)."""
    try:
        shape = tuple(a.shape)
    except Exception:
        shape = "<unknown-shape>"

    try:
        dtype = getattr(a, "dtype", None)
        dtype = jnp.dtype(dtype).name if dtype is not None else "<unknown-dtype>"
    except Exception:
        dtype = "<unknown-dtype>"

    wt = getattr(a, "weak_type", None)
    if wt is None:
        wt = getattr(getattr(a, "aval", None), "weak_type", None)
    wt = bool(wt) if wt is not None else False

    # These should not materialize values
    dev = None
    try:
        d = a.device()
        dev = str(d)
    except Exception:
        dev = None

    sharding = None
    try:
        sh = getattr(a, "sharding", None)
        sharding = str(sh) if sh is not None else None
    except Exception:
        sharding = None

    return ("jax.Array", dtype, shape, wt, dev, sharding)


def _fingerprint(obj):
    """
    Donation-safe fingerprint:
      - JAX arrays: use metadata only (no repr/value)
      - hashables: use hash+short safe repr
      - unhashables: use type + id + short safe repr (repr guarded)
    """
    if _is_jax_array(obj):
        return _safe_array_fingerprint(obj)

    try:
        h = hash(obj)
        return ("hashable", type(obj).__name__, h, _safe_repr(obj))
    except Exception:
        return ("unhashable", type(obj).__name__, id(obj), _safe_repr(obj))


def _pretty_fp(fp):
    """Pretty-print a fingerprint for logs."""
    if fp is None:
        return "<missing>"
    if isinstance(fp, tuple) and fp and fp[0] == "jax.Array":
        _, dtype, shape, wt, dev, sharding = fp
        wt_s = ", weak_type=True" if wt else ""
        dev_s = f", device={dev}" if dev else ""
        sh_s = f", sharding={sharding}" if sharding else ""
        return f"ShapedArray({dtype}{list(shape) if shape != () else ''}{wt_s}){dev_s}{sh_s}"

    if fp[0] == "hashable":
        # ("hashable", type, hash, repr)
        return fp[3]
    # ("unhashable", type, id, repr)
    return f"{fp[3]} (id={fp[2]})"


# ----------------------------
# Dynamic signature (cache key-ish)
# ----------------------------
def _path_to_suffix(path):
    parts = []
    for key in path:
        if hasattr(jax.tree_util, "GetAttrKey") and isinstance(key, jax.tree_util.GetAttrKey):
            parts.append(f".{key.name}")
        elif hasattr(jax.tree_util, "SequenceKey") and isinstance(key, jax.tree_util.SequenceKey):
            parts.append(f"[{key.idx}]")
        elif hasattr(jax.tree_util, "DictKey") and isinstance(key, jax.tree_util.DictKey):
            parts.append(f"[{key.key!r}]")
        else:
            parts.append(f"[{key}]")
    return "".join(parts)


def _dyn_leaf_sig(leaf):
    """
    Better approximation of JAX cache key leaf signature:
      - shape
      - dtype
      - weak_type (for Python scalars AND some JAX 0-d arrays)
      - device/sharding (when available)
    """
    if isinstance(leaf, (int, float, bool)):
        dtype = jnp.result_type(leaf).name
        return ("py.scalar", dtype, (), True, None, None)

    if _is_jax_array(leaf):
        # use the same array fingerprint data structure
        fp = _safe_array_fingerprint(leaf)
        _, dtype, shape, wt, dev, sharding = fp
        return ("jax.array", dtype, shape, wt, dev, sharding)

    # numpy arrays etc.
    shape = tuple(jnp.shape(leaf))
    dtype = jnp.result_type(leaf).name
    return ("other", dtype, shape, False, None, None)


def dynamic_signature_kv(bound_arguments, static_argnames):
    """
    Return a dict mapping leaf path -> signature tuple.
    """
    static_argnames = set(static_argnames)
    kv = {}
    for name, value in bound_arguments.items():
        if name in static_argnames:
            continue
        for path, leaf in jax.tree_util.tree_leaves_with_path(value):
            full = name + _path_to_suffix(path)
            kv[full] = _dyn_leaf_sig(leaf)
    return kv


def diff_kv(prev_kv, cur_kv):
    changes = []
    all_keys = set(prev_kv) | set(cur_kv)
    for k in sorted(all_keys):
        a = prev_kv.get(k, "<missing>")
        b = cur_kv.get(k, "<missing>")
        if a != b:
            changes.append((k, a, b))
    return changes


# ----------------------------
# “Other python args” fingerprinting
# ----------------------------
def contains_jax_array(x) -> bool:
    try:
        leaves = jax.tree_util.tree_leaves(x)
        return any(_is_jax_array(l) for l in leaves)
    except Exception:
        return _is_jax_array(x)


def other_python_fps(bound_args, static_argnames):
    """
    Fingerprint only non-array-ish objects that do NOT contain arrays,
    to avoid donation crashes from repr() on pytrees with arrays (e.g. buffer_state).
    """
    out = {}
    for k, v in bound_args.items():
        if k in static_argnames:
            continue
        if not contains_jax_array(v):
            out[k] = _fingerprint(v)
    return out


# ----------------------------
# Main monitor
# ----------------------------
def monitor_recompiles(jit_fn, static_argnames=None, max_diffs=30):
    """
    Wrap a jitted function; on each new compilation:
      - If static signature is new: print only the (minimal) changed static args, plus all prior cached values.
      - If static signature already cached: diff dynamic signature and non-array args vs closest cached variant.

    Safe with donate_argnames/donate_argnums.
    """
    if static_argnames is None:
        static_argnames = tuple(infer_static_argnames(jit_fn))
    static_argnames = tuple(static_argnames)

    py_fn = getattr(jit_fn, "__wrapped__", jit_fn)
    sig = inspect.signature(py_fn)

    if not hasattr(jit_fn, "_cache_size"):
        raise RuntimeError("jit_fn has no _cache_size(); can't detect recompiles in this JAX build.")

    variants = []  # each: {"static_key":..., "static_fp":..., "dyn_kv":..., "other_fp":...}
    seen_static_keys = set()

    def static_fp_from(bound_args):
        return {k: _fingerprint(bound_args[k]) for k in static_argnames if k in bound_args}

    def static_key_of(sfp):
        return tuple(sorted(sfp.items()))

    def all_cached_values_for(name):
        seen = set()
        out = []
        for v in variants:
            fp = v["static_fp"].get(name, None)
            if fp is None:
                continue
            if fp not in seen:
                out.append(fp)
                seen.add(fp)
        return out

    @functools.wraps(py_fn)
    def wrapped(*args, **kwargs):
        before = jit_fn._cache_size()
        out = jit_fn(*args, **kwargs)
        after = jit_fn._cache_size()

        if after > before:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            bargs = bound.arguments

            sfp = static_fp_from(bargs)
            skey = static_key_of(sfp)
            dyn_kv = dynamic_signature_kv(bargs, static_argnames)
            oth_fp = other_python_fps(bargs, static_argnames)

            print(f"[jit recompile] cache_size {before} -> {after}")

            if not variants:
                print("  first compilation.")
            else:
                if skey in seen_static_keys:
                    print("  static signature already cached; investigating deeper...")

                    same_static = [v for v in variants if v["static_key"] == skey]
                    if not same_static:
                        print("  (no prior variant found with same static signature; unexpected)")
                    else:
                        # pick best match by minimal dynamic diffs
                        best = None
                        best_diffs = None
                        for v in same_static:
                            diffs = diff_kv(v["dyn_kv"], dyn_kv)
                            if best_diffs is None or len(diffs) < len(best_diffs):
                                best, best_diffs = v, diffs

                        if best_diffs:
                            print(f"  dynamic leaf diffs (showing up to {max_diffs}):")
                            for name, a, b in best_diffs[:max_diffs]:
                                print(f"    {name}: {a} -> {b}")
                            if len(best_diffs) > max_diffs:
                                print(f"    ... {len(best_diffs) - max_diffs} more")
                        else:
                            print("  dynamic signature (shape/dtype/weak_type/device/sharding) matches cached variant.")

                        other_diffs = diff_kv(best["other_fp"], oth_fp)
                        if other_diffs:
                            print(f"  non-array argument diffs (likely missing static_argnames; showing up to {max_diffs}):")
                            for name, a, b in other_diffs[:max_diffs]:
                                print(f"    {name}: {_pretty_fp(a)} -> {_pretty_fp(b)}")
                            if len(other_diffs) > max_diffs:
                                print(f"    ... {len(other_diffs) - max_diffs} more")
                        else:
                            print("  non-array arguments also match.")
                            print("  if still recompiling, next suspects are:")
                            print("    - different device/backend between calls")
                            print("    - different compilation options / JAX config changes")
                            print("    - missed static arg that *contains* arrays but is used as static via closure/global")
                else:
                    # New static signature: find minimal static diffs vs closest previous
                    best_changed = None
                    for v in variants:
                        prev = v["static_fp"]
                        changed = [n for n in sfp.keys() if prev.get(n) != sfp.get(n)]
                        if best_changed is None or len(changed) < len(best_changed):
                            best_changed = changed

                    if best_changed:
                        print("  static args with new value(s) contributing to this compile:")
                        for n in best_changed:
                            prev_vals = all_cached_values_for(n)
                            prev_vals_str = ", ".join(_pretty_fp(fp) for fp in prev_vals) if prev_vals else "<none>"
                            print(f"    {n} = {bargs[n]!r}")
                            print(f"    {n} previously cached values: {prev_vals_str}")
                    else:
                        print("  (no static args inferred or no static differences found)")

            variants.append({"static_key": skey, "static_fp": sfp, "dyn_kv": dyn_kv, "other_fp": oth_fp})
            seen_static_keys.add(skey)

        return out

    return wrapped


# ----------------------------
# Optional: quick demo
# ----------------------------
if __name__ == "__main__":
    from flax import struct
    from functools import partial

    @struct.dataclass
    class Data:
        scale: int
        value: jax.Array

    @partial(jax.jit, static_argnames=["z"])
    def f(x: Data, y: float, z: int):
        return (x.value + y) * x.scale if z else (x.value - y) * x.scale

    x = Data(scale=2, value=jnp.ones(2))
    y = 2.0

    f_mon = monitor_recompiles(f)
    f_mon(x, y, 1)
    f_mon(x, y, 2)
    f_mon(x, y, 0)
    f_mon(x, y, 1)
