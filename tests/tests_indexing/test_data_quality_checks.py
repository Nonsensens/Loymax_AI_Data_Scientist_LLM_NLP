import pandas as pd
from indexing.indexingService import data_quality_checks
import os

def test_data_quality_filters():
    os.environ["MIN_TEXT_LENGTH"] = "10"
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "text": ["тест1", "тест1", "достаточно длинный текст"]
    })
    result = data_quality_checks(df)
    assert len(result) == 1
    assert "достаточно длинный текст" in result.iloc[0]["text"]
