"""Test TileDB functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores import TileDB
from tests.integration_tests.vectorstores.fake_embeddings import ConsistentFakeEmbeddings

def test_tiledb(tmp_path) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = TileDB.from_texts(texts=texts, embedding=ConsistentFakeEmbeddings(), array_uri=str(tmp_path))
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_tiledb_updates(tmp_path) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    ids = [1, 2, 3]
    docsearch = TileDB.from_texts(texts=texts, embedding=ConsistentFakeEmbeddings(), array_uri=str(tmp_path), ids=ids)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    docsearch.delete([1, 3])
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="bar")]
    output = docsearch.similarity_search("baz", k=1)
    assert output == [Document(page_content="bar")]

    docsearch.add_texts(texts=["fooo", "bazz"], ids=[4, 5])
    output = docsearch.similarity_search("fooo", k=1)
    assert output == [Document(page_content="fooo")]
    output = docsearch.similarity_search("bazz", k=1)
    assert output == [Document(page_content="bazz")]

    docsearch.consolidate_updates()
    output = docsearch.similarity_search("fooo", k=1)
    assert output == [Document(page_content="fooo")]
    output = docsearch.similarity_search("bazz", k=1)
    assert output == [Document(page_content="bazz")]
