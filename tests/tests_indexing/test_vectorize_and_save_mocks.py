from unittest.mock import MagicMock, patch
import pandas as pd
import os
from indexing.indexingService import vectorize_and_save

@patch("indexing.indexingService.Chroma")
@patch("indexing.indexingService.HuggingFaceEmbeddings")
def test_vectorize_and_save_mocks(mock_embed, mock_chroma):
    mock_chroma.return_value.get.return_value = {"documents": []}
    df = pd.DataFrame({"text": ["тест документ"]})
    os.environ["CHROMA_DB_DIR"] = "./vector_db"
    os.environ["CHUNK_SIZE"] = "100"
    os.environ["CHUNK_OVERLAP"] = "20"
    os.environ["EMBEDDING_MODEL"] = "mock-model"
    db, retriever = vectorize_and_save(df)
    assert mock_chroma.called
