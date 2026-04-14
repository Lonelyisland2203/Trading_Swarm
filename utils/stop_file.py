"""Stop file (kill switch) utilities."""

from pathlib import Path


class StopFileChecker:
    """
    Check for STOP file existence (trading kill switch).

    The STOP file is a simple file-based mechanism to halt trading:
    - Create the file: trading stops
    - Remove the file: trading can resume

    Default path is execution/state/STOP.
    """

    DEFAULT_PATH = Path("execution/state/STOP")

    def __init__(self, path: Path | None = None):
        """
        Initialize with optional custom path.

        Args:
            path: Custom STOP file path, or None to use DEFAULT_PATH
        """
        self._path = path or self.DEFAULT_PATH

    @property
    def path(self) -> Path:
        """Get the STOP file path."""
        return self._path

    def is_active(self) -> bool:
        """
        Check if STOP file exists.

        Returns:
            True if STOP file exists (trading should halt)
        """
        return self._path.exists()

    def create(self) -> None:
        """Create the STOP file to halt trading."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch()

    def remove(self) -> None:
        """Remove the STOP file to allow trading."""
        if self._path.exists():
            self._path.unlink()


# Default instance for common usage
default_stop_checker = StopFileChecker()
