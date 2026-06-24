"""Storage skeleton for benchmark artifacts."""

from pathlib import Path


class LocalArtifactStore:
    """Small local store for reports and trace exports."""

    def __init__(self, root: Path = Path("reports")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_text(self, relative_path: str, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


def save_text(path: str, content: str) -> None:
    """Write content to an arbitrary path, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
