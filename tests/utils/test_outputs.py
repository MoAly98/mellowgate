"""Tests for mellowgate.utils.outputs module."""

import os
import tempfile
from pathlib import Path

from mellowgate.utils.outputs import OutputManager


class TestOutputManager:
    """Test the OutputManager class."""

    def test_output_manager_creation_default(self):
        """Test creating OutputManager with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            assert manager.base_directory == Path(temp_dir).resolve()

    def test_output_manager_creation_custom(self):
        """Test creating OutputManager with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = os.path.join(temp_dir, "custom_outputs")
            manager = OutputManager(base_directory=custom_dir)

            assert manager.base_directory == Path(custom_dir).resolve()
            # Directory should be created automatically
            assert manager.base_directory.exists()

    def test_output_manager_string_path(self):
        """Test OutputManager with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            assert isinstance(manager.base_directory, Path)
            assert str(manager.base_directory) == str(Path(temp_dir).resolve())

    def test_output_manager_path_object(self):
        """Test OutputManager with Path object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            assert isinstance(manager.base_directory, Path)

    def test_get_path_directory_only(self):
        """Test getting directory path without filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            output_path = manager.get_path("test_dir")
            expected_path = (Path(temp_dir) / "test_dir").resolve()

            assert output_path == expected_path
            # Directory should be created automatically
            assert output_path.exists()
            assert output_path.is_dir()

    def test_get_path_with_filename(self):
        """Test getting path with filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            output_path = manager.get_path("plots", "figure.png")
            expected_path = (Path(temp_dir) / "plots" / "figure.png").resolve()

            assert output_path == expected_path
            # Parent directory should be created automatically
            assert output_path.parent.exists()
            assert output_path.parent.is_dir()

    def test_get_path_nested_subdirectories(self):
        """Test getting path with nested subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Use nested path as subdirectory
            output_path = manager.get_path("results/experiment1", "data.csv")
            expected_path = (
                Path(temp_dir) / "results" / "experiment1" / "data.csv"
            ).resolve()

            assert output_path == expected_path
            # Parent directory should be created automatically
            assert output_path.parent.exists()

    def test_directory_creation_automatic(self):
        """Test that directories are created automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Get path for new subdirectory
            output_path = manager.get_path("new_dir")

            # Directory should have been created
            assert output_path.exists()
            assert output_path.is_dir()

    def test_directory_creation_nested(self):
        """Test creation of nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Get path with nested structure
            output_path = manager.get_path("a/b/c", "file.txt")

            # All parent directories should be created
            assert output_path.parent.exists()
            assert (Path(temp_dir) / "a").exists()
            assert (Path(temp_dir) / "a" / "b").exists()
            assert (Path(temp_dir) / "a" / "b" / "c").exists()

    def test_output_manager_workflow(self):
        """Test typical OutputManager workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Get paths for different types of outputs
            plot_path = manager.get_path("plots", "experiment.png")
            data_path = manager.get_path("data", "results.csv")

            # Check that directories were created
            assert plot_path.parent.exists()
            assert data_path.parent.exists()

            # Check expected structure
            assert (Path(temp_dir) / "plots").exists()
            assert (Path(temp_dir) / "data").exists()

    def test_output_manager_absolute_paths(self):
        """Test that OutputManager returns absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            output_path = manager.get_path("test_dir", "file.txt")

            assert output_path.is_absolute()

    def test_output_manager_special_characters(self):
        """Test OutputManager with special characters in paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Test with spaces and special characters
            output_path = manager.get_path("test dir", "file with spaces.txt")
            expected_path = (
                Path(temp_dir) / "test dir" / "file with spaces.txt"
            ).resolve()

            assert output_path == expected_path
            assert output_path.parent.exists()

    def test_multiple_calls_same_directory(self):
        """Test multiple calls for the same directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Call get_path multiple times for same directory
            path1 = manager.get_path("shared_dir", "file1.txt")
            path2 = manager.get_path("shared_dir", "file2.txt")

            # Should point to same parent directory
            assert path1.parent == path2.parent
            assert path1.parent.exists()

            # Files should have different names
            assert path1.name == "file1.txt"
            assert path2.name == "file2.txt"


class TestOutputManagerEdgeCases:
    """Test edge cases for OutputManager."""

    def test_output_manager_nonexistent_base_directory(self):
        """Test OutputManager with nonexistent base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = os.path.join(temp_dir, "does_not_exist_yet")

            # Should create manager and directory
            manager = OutputManager(base_directory=nonexistent_dir)

            assert manager.base_directory == Path(nonexistent_dir).resolve()
            assert manager.base_directory.exists()

    def test_get_path_no_filename(self):
        """Test get_path with no filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Just the subdirectory
            output_path = manager.get_path("test_subdir")

            assert output_path == (Path(temp_dir) / "test_subdir").resolve()
            assert output_path.exists()

    def test_get_path_with_filename(self):
        """Test get_path with filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            output_path = manager.get_path("test_subdir", "single_file.txt")
            expected_path = (
                Path(temp_dir) / "test_subdir" / "single_file.txt"
            ).resolve()

            assert output_path == expected_path
            assert output_path.parent.exists()

    def test_relative_base_directory(self):
        """Test OutputManager with relative base directory."""
        # Use a relative path
        relative_dir = "test_output_rel"

        manager = OutputManager(base_directory=relative_dir)

        # Should convert to absolute path
        assert manager.base_directory.is_absolute()
        assert manager.base_directory.name == "test_output_rel"

        # Clean up
        if manager.base_directory.exists():
            import shutil

            shutil.rmtree(manager.base_directory)

    def test_path_normalization(self):
        """Test that paths are properly normalized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use path with extra separators
            manager = OutputManager(base_directory=temp_dir)

            output_path = manager.get_path(
                "subdir//with//extra//separators", "file.txt"
            )

            # Path should work correctly
            assert output_path.parent.exists()
            assert "file.txt" in str(output_path)


class TestOutputManagerIntegration:
    """Integration tests for OutputManager."""

    def test_output_manager_real_file_operations(self):
        """Test OutputManager with actual file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Get output path for a file
            output_path = manager.get_path("experiments", "test_data.txt")

            # Write a test file
            with open(output_path, "w") as f:
                f.write("test data")

            # Verify file was created
            assert output_path.exists()
            assert output_path.is_file()

            # Verify content
            with open(output_path, "r") as f:
                content = f.read()
                assert content == "test data"

    def test_output_manager_multiple_files_same_directory(self):
        """Test creating multiple files in the same directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Create multiple files in same directory
            files = ["file1.txt", "file2.txt", "file3.txt"]
            output_paths = []

            for filename in files:
                path = manager.get_path("shared_dir", filename)
                output_paths.append(path)

                # Create the file
                with open(path, "w") as f:
                    f.write(f"content of {filename}")

            # Verify all files exist
            for path in output_paths:
                assert path.exists()
                assert path.is_file()

            # Verify they're all in the same directory
            parent_dirs = [path.parent for path in output_paths]
            assert len(set(parent_dirs)) == 1  # All have same parent

    def test_output_manager_hierarchical_structure(self):
        """Test creating hierarchical directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OutputManager(base_directory=temp_dir)

            # Create hierarchical structure
            structure = [
                ("experiments/exp1", "data.csv"),
                ("experiments/exp1/plots", "figure1.png"),
                ("experiments/exp2", "data.csv"),
                ("experiments/exp2/plots", "figure1.png"),
                ("results", "summary.txt"),
            ]

            created_paths = []
            for subdir, filename in structure:
                path = manager.get_path(subdir, filename)
                created_paths.append(path)

                # Create the file
                with open(path, "w") as f:
                    f.write(f"content for {subdir}/{filename}")

            # Verify directory structure
            base_path = Path(temp_dir)
            assert (base_path / "experiments").exists()
            assert (base_path / "experiments" / "exp1").exists()
            assert (base_path / "experiments" / "exp1" / "plots").exists()
            assert (base_path / "experiments" / "exp2").exists()
            assert (base_path / "experiments" / "exp2" / "plots").exists()
            assert (base_path / "results").exists()

            # Verify all files exist
            for path in created_paths:
                assert path.exists()
                assert path.is_file()

    def test_default_outputs_directory(self):
        """Test OutputManager with default 'outputs' directory."""
        # Test the default behavior
        manager = OutputManager()  # Uses default "outputs"

        assert manager.base_directory.name == "outputs"
        assert manager.base_directory.exists()

        # Get a path within it
        test_path = manager.get_path("test", "file.txt")
        assert "outputs" in str(test_path)
        assert test_path.parent.exists()

        # Clean up
        import shutil

        if manager.base_directory.exists():
            shutil.rmtree(manager.base_directory)
