import pytest
from czbenchmarks.datasets.types import Organism


class TestOrganism:
    """Tests for Organism Enum."""

    def test_enum_properties(self):
        """Test that Organism enum properties return correct values."""
        organism = Organism.HUMAN

        assert organism.name == "homo_sapiens"
        assert organism.prefix == "ENSG"

    def test_invalid_organism(self):
        """Test that accessing an invalid organism raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = Organism.INVALID

    def test_enum_representation(self):
        """Test string and repr representation of Organism enum."""
        organism = Organism.MOUSE

        assert str(organism) == "mus_musculus"
        assert repr(organism) == "mus_musculus"
