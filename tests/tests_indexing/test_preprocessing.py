import pytest
from utils.utils import preprocess_text

@pytest.mark.parametrize("input_text,expected", [
    (" Hello World ", "hello world"),
    ("TEST", "test"),
    ("   ", ""),
    ("123TEXT", "123text")
])
def test_preprocess_text(input_text, expected):
    assert preprocess_text(input_text) == expected
