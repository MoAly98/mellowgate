import os
from pathlib import Path
from typing import Optional

class OutputManager:
    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, subdir: str, filename: Optional[str] = None) -> Path:
        path = self.base_dir
        path = path / Path(subdir)
        path.mkdir(parents=True, exist_ok=True)
        if filename:
            return path / filename
        return path