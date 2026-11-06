"""The verification gate (pre-registered rule, report, full-call amendment) and its input checks."""

from .analysis import GateBlocked, GateSpec, analyze, boots, compare, compile_summary, default_unit, task_means, verdict_lines, write_report
from .fullcall import amend, summary as fullcall_summary
from .io import load_arm, load_study_calls, read_jsonl_unique, universe

__all__ = ["GateBlocked", "GateSpec", "amend", "analyze", "boots", "compare", "compile_summary", "default_unit",
           "fullcall_summary", "load_arm", "load_study_calls", "read_jsonl_unique", "task_means", "universe",
           "verdict_lines", "write_report"]


def prereg_template() -> str:
    """The pre-registration template (markdown) shipped with the package."""
    from importlib.resources import files

    return (files(__package__) / "prereg_template.md").read_text()
