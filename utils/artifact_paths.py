from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_artifact_path(value, summary_path=None):
    """Resolve project-relative paths, then legacy summary-relative paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [PROJECT_ROOT / path]
    if summary_path is not None:
        candidates.append(Path(summary_path).resolve().parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def portable_artifact_path(value):
    """Store in-project artifacts relative to the project root."""
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
