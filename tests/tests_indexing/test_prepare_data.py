import pandas as pd
from indexing.indexingService import prepare_data

def test_prepare_data_filters_and_cleans_text():
    df = pd.DataFrame({"text": ["Тест", " ", "\n", "Тест"]})
    cleaned_df = prepare_data(df)
    assert len(cleaned_df) == 1
    print(cleaned_df.iloc[0]["text"])
    assert cleaned_df.iloc[0]["text"] == "тест"
