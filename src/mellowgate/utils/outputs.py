"""Output management utilities for organizing experimental results.

This module provides tools for managing file and directory structures when
running experiments. The OutputManager class helps create consistent directory
hierarchies and generates appropriate file paths for saving results, logs,
and other experimental artifacts.

The utilities ensure reproducible organization of experimental outputs and
make it easy to locate and manage results from multiple runs.
"""

from pathlib import Path
from typing import Optional


class OutputManager:
    """Manages output directories and file paths for experimental results.

    This class provides a centralized way to organize and manage output files
    from experiments, ensuring consistent directory structure and easy access
    to generated content like plots, logs, and data files.

    Attributes:
        base_directory: The root directory where all outputs are stored.

    Examples:
        >>> output_manager = OutputManager("my_experiment_outputs")
        >>> plot_path = output_manager.get_path("plots", "results.png")
        >>> data_path = output_manager.get_path("data")
    """

    def __init__(self, base_directory: str = "outputs") -> None:
        """Initialize the OutputManager with a base directory.

        Args:
            base_directory: The root directory name for all outputs.
                           Defaults to "outputs". Will be created if it
                           doesn't exist.
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def get_path(self, subdirectory: str, filename: Optional[str] = None) -> Path:
        """Get a path within the output directory structure.

        Creates the subdirectory if it doesn't exist and optionally returns
        the full path to a specific file within that subdirectory.

        Args:
            subdirectory: The subdirectory name within the base directory.
                         Will be created if it doesn't exist.
            filename: Optional filename to append to the path. If not provided,
                     returns the path to the subdirectory itself.

        Returns:
            pathlib.Path: The full path to either the subdirectory or the
                         specified file within the subdirectory.

        Examples:
            >>> output_manager = OutputManager("results")
            >>> # Get directory path
            >>> dir_path = output_manager.get_path("plots")
            >>> # Get file path
            >>> file_path = output_manager.get_path("plots", "figure1.png")
        """
        target_path = self.base_directory / Path(subdirectory)
        target_path.mkdir(parents=True, exist_ok=True)

        if filename:
            return target_path / filename
        return target_path
