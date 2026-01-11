import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Warp DeprecationWarning:.*")
warnings.filterwarnings("ignore", module=r"^warp(\.|$)")
import warp._src.utils as wu
wu.warn = lambda *a, **k: None