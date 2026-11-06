"""The proposer kit: forward-pass budget server, proposer-side weight kernel, brief template, packet builder,
macOS sandbox launchers (Claude Code, Codex) and the transcript audit."""

from importlib.resources import files as _files


def sandbox_dir():
    """Directory of the shipped launchers and Seatbelt profiles (macOS only)."""
    return _files(__package__) / "sandbox"
