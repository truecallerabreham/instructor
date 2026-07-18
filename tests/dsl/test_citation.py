"""Regression test for CitationMixin crash when context missing 'context' key.

See: https://github.com/567-labs/instructor/issues/2459
"""
import pytest
from pydantic import BaseModel, Field

from instructor import CitationMixin


class CitationModel(CitationMixin, BaseModel):
    answer: str = Field(description="The answer")
    substring_quotes: list[str] = Field(
        default_factory=list,
        description="Quotes supporting the answer",
    )


class TestCitationMixinNoneContext:
    def test_context_missing_context_key_does_not_crash(self):
        """validate_sources should not crash when context dict lacks 'context' key."""
        model = CitationModel.model_validate(
            {"answer": "test", "substring_quotes": ["test"]},
            context={"foo": "bar"},
        )
        assert model.answer == "test"

    def test_context_is_none_does_not_crash(self):
        """validate_sources should handle None context gracefully."""
        model = CitationModel.model_validate(
            {"answer": "test", "substring_quotes": ["test"]},
        )
        assert model.answer == "test"

    def test_context_with_context_key_works_normally(self):
        """validate_sources should work normally when context key is present."""
        context = "Jason was a student. Jason is 20 years old."
        model = CitationModel.model_validate(
            {"answer": "Jason", "substring_quotes": ["Jason"]},
            context={"context": context},
        )
        assert model.answer == "Jason"
        assert len(model.substring_quotes) >= 1
