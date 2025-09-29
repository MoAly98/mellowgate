"""Simplified unit tests for utils.outputs module.

This module provides essential tests for output management utilities.
"""

import tempfile
from pathlib import Path

from mellowgate.utils.outputs import OutputManager


class TestOutputManager:
    """Test OutputManager class for managing output directories and files."""

    def test_init_and_basic_functionality(self):
        """Test OutputManager initialization and basic directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=str(Path(temp_dir) / "test_outputs"))

            # Check directory was created
            assert manager.base_directory.exists()
            assert manager.base_directory.is_dir()

    def test_get_path_subdirectory(self):
        """Test getting path to subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=str(Path(temp_dir) / "test_outputs"))

            subdir_path = manager.get_path("plots")

            # Check subdirectory was created and path is correct
            assert subdir_path.exists()
            assert subdir_path.is_dir()
            assert subdir_path.name == "plots"

    def test_get_path_with_filename(self):
        """Test getting path to file within subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=str(Path(temp_dir) / "test_outputs"))

            file_path = manager.get_path("data", "results.csv")

            # Check path structure and subdirectory creation
            assert file_path.parent.exists()  # Subdirectory created
            assert file_path.name == "results.csv"
            assert not file_path.exists()  # File itself not created
