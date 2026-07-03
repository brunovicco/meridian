"""Loader for a compiled (optimised) DSPy program.

DSPy's value compounds when a program is *compiled* against a training set: an
optimiser tunes the prompts and few-shot demonstrations, and the result is saved
as a JSON artifact. Loading that artifact at startup means the router runs the
optimised program rather than a cold one, and upgrading the model provider
becomes "recompile and reload", not "hand-edit prompts".

This loader is the adoption point. If a compiled artifact is present it's
loaded onto the module; if not, the module runs uncompiled (still correct, just
not optimised). Keeping this seam explicit is what makes the compile/reload
workflow a first-class operation rather than an afterthought.
"""

from pathlib import Path
from typing import Any

from meridian.infrastructure.dspy.grok import dspy_available


def load_compiled_program(module: Any, artifact_path: Path) -> bool:
    """Load a compiled DSPy program onto ``module`` if the artifact exists.

    :param module: A DSPy module instance exposing ``load`` (the dspy API).
    :param artifact_path: Path to the compiled program JSON.
    :returns: ``True`` if a compiled program was loaded, ``False`` otherwise.
    """
    if not dspy_available():
        return False
    if not artifact_path.exists():
        return False
    try:  # pragma: no cover - requires dspy and a real artifact
        module.load(str(artifact_path))
        return True
    except Exception:  # noqa: BLE001 - a bad artifact must not break startup
        return False
