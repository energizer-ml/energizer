from .compiler.bouncer import Bouncer, BouncerConfig
from . import runner
from .monitor import monitor


def compile(*args, **kwargs):
    try:
        from .compiler.transpiler import compile_to_coreml
    except ModuleNotFoundError as exc:
        missing = exc.name or "coremltools"
        raise ModuleNotFoundError(
            f"{missing} is required for energizer.backpack.compile()"
        ) from exc
    return compile_to_coreml(*args, **kwargs)
