"""Tests for mellowgate.core.branch module."""

from mellowgate.core import Bound, Branch


class TestBound:
    """Test the Bound dataclass."""

    def test_bound_creation_default(self):
        """Test creating a Bound with default parameters."""
        bound = Bound(value=1.0)
        assert bound.value == 1.0
        assert bound.inclusive is True

    def test_bound_creation_exclusive(self):
        """Test creating a Bound with explicit parameters."""
        bound = Bound(value=2.5, inclusive=False)
        assert bound.value == 2.5
        assert bound.inclusive is False


class TestBranch:
    """Test the Branch dataclass."""

    def test_branch_creation_minimal(self):
        """Test creating a Branch with minimal parameters."""

        def func(th):
            return th**2

        branch = Branch(function=func)
        assert branch.function is func
        assert branch.derivative_function is None
        assert branch.threshold == (None, None)

    def test_branch_with_derivative(self):
        """Test creating a Branch with derivative function."""

        def func(th):
            return th**2

        def deriv(th):
            return 2 * th

        branch = Branch(function=func, derivative_function=deriv)
        assert branch.function is func
        assert branch.derivative_function is deriv

    def test_branch_with_threshold(self):
        """Test creating a Branch with threshold bounds."""

        def func(th):
            return th**2

        threshold = (Bound(0, inclusive=True), Bound(5, inclusive=False))
        branch = Branch(function=func, threshold=threshold)
        assert branch.threshold == threshold
