#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete local wandb/ckpts and optionally purge a W&B project."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform deletions (required).",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Skip local deletion.",
    )
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Skip cloud deletion.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (user or team). Can also use WANDB_ENTITY.",
    )
    parser.add_argument(
        "--project",
        default="jaxrl",
        help="W&B project name (default: jaxrl).",
    )
    return parser.parse_args()


def infer_entity():
    env_entity = __import__("os").environ.get("WANDB_ENTITY")
    if env_entity:
        return env_entity

    try:
        import wandb
        api = wandb.Api()
        if hasattr(api, "default_entity") and api.default_entity:
            return api.default_entity
    except Exception:
        pass

    settings_path = Path.home() / ".config" / "wandb" / "settings"
    if settings_path.exists():
        for line in settings_path.read_text().splitlines():
            if line.strip().startswith("entity"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    value = parts[1].strip().strip('"').strip("'")
                    if value:
                        return value

    return None


def delete_local(repo_root: Path):
    targets = [repo_root / "wandb", repo_root / "ckpts"]
    total_files = 0
    for path in targets:
        if path.exists():
            total_files += sum(1 for p in path.rglob("*") if p.is_file())
            for child in path.iterdir():
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            print(f"Emptied local: {path}")
        else:
            print(f"Skip (missing): {path}")
    print(f"Local files deleted: {total_files}")


def list_cloud_runs(entity: str, project: str):
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "wandb is not available. Install it or skip cloud deletion."
        ) from exc

    api = wandb.Api()
    return list(api.runs(f"{entity}/{project}"))


def delete_cloud(runs):
    print(f"Cloud runs to delete: {len(runs)}")
    for run in runs:
        run.delete()
        print(f"Deleted cloud run: {run.id}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    do_local = not args.no_local
    do_cloud = not args.no_cloud

    local_counts = []
    if do_local:
        for path in (repo_root / "wandb", repo_root / "ckpts"):
            if path.exists():
                count = sum(1 for p in path.rglob("*") if p.is_file())
                local_counts.append(f"{path.name}/: {count} files")
            else:
                local_counts.append(f"{path.name}/: missing")

    runs = None
    entity = None
    if do_cloud:
        entity = args.entity or infer_entity()
        if not entity:
            raise RuntimeError("Missing W&B entity. Use --entity or WANDB_ENTITY.")
        runs = list_cloud_runs(entity, args.project)

    if not args.yes:
        print("Dry run. Add --yes to perform deletions.")
        if do_local:
            print("Would delete local: " + ", ".join(local_counts))
        if do_cloud:
            print(f"Would delete cloud runs: {len(runs)} ({entity}/{args.project})")
        return 1

    print("About to delete data. Double check before proceeding.")
    if do_local:
        print("Local targets: " + ", ".join(local_counts))
    if do_cloud:
        print(f"Cloud runs: {len(runs)} ({entity}/{args.project})")
    confirm = input('Type "DELETE" to continue: ').strip()
    if confirm != "DELETE":
        print("Aborted.")
        return 1

    if do_local:
        delete_local(repo_root)

    if do_cloud:
        delete_cloud(runs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
