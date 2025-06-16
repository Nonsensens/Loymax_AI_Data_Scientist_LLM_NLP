import pandas as pd
import tempfile
import os
import pytest

from indexing.indexingService import load_data

def test_load_data_raises_without_text_column():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("not_text\nsome_value")
        f.close()
        os.environ["DATA_PATH"] = f.name
        with pytest.raises(ValueError):
            load_data()
