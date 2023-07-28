"""Test TileDB functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores import TileDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

def test_tiledb(tmp_path) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = TileDB.from_texts(texts=texts, embedding=FakeEmbeddings(), array_uri=str(tmp_path))
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]
