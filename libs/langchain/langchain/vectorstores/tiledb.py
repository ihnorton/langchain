"""Wrapper around TileDB vector database."""
from __future__ import annotations

import os
import pickle
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from ratelimiter import RateLimiter

import numpy as np
import tiledb

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

INDEX_METRICS = frozenset(["euclidean"])
DEFAULT_METRIC = "euclidean"
DOCUMENTS_ARRAY_NAME = "documents"
VECTOR_ARRAY_NAME = "vectors"


def dependable_tiledb_import() -> Any:
    """Import tiledb-vector-search if available, otherwise raise error."""
    try:
        import tiledb.vector_search as tiledb_vs
    except ImportError:
        raise ValueError(
            "Could not import tiledb-vector-search python package. "
            "Please install it with `conda install -c tiledb tiledb-vector-search` "
            "or `pip install tiledb-vector-search`"
        )
    return tiledb_vs


class TileDB(VectorStore):
    """Wrapper around TileDB vector database.

    To use, you should have the ``tiledb-vector-search`` python package installed.

    Example:
        .. code-block:: python

            from langchain import TileDB
            db = TileDB(embedding_function, index, metric, docs_array)

    """

    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        metric: str,
        docs_array: tiledb.Array,
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index = index
        self.metric = metric
        self.docs_array = docs_array

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Accept embedding object directly
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError(
            "TileDB does not allow to add new data once the index is built."
        )

    def process_index_results(self, idxs: List[int]) -> List[Tuple[Document, float]]:
        """Turns TileDB results into a list of documents and scores.

        Args:
            idxs: List of indices of the documents in the index.
            dists: List of distances of the documents in the index.
        Returns:
            List of Documents and scores.
        """
        docs = []
        for idx in idxs[0]:
            doc = self.docs_array[idx]
            if doc is None:
                raise ValueError(f"Could not find document for id {idx}, got {doc}")
            pickled_metadata = doc.get("metadata")
            if pickled_metadata is not None:
                metadata = pickle.loads(
                    np.array(pickled_metadata.tolist()).astype(np.uint8).tobytes()
                )
                docs.append(
                    (Document(page_content=str(doc["text"]), metadata=metadata), 1.0)
                )
            else:
                docs.append(
                    (Document(page_content=str(doc["text"])), 1.0)
                )
        return docs

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, search_k: int = -1
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided
        Returns:
            List of Documents most similar to the query and score for each
        """
        idxs = self.index.query(
            np.array([np.array(embedding).astype(np.float32)]).astype(np.float32), k=k
        )
        return self.process_index_results(idxs)

    def similarity_search_with_score_by_index(
        self, docstore_index: int, k: int = 4, search_k: int = -1
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided
        Returns:
            List of Documents most similar to the query and score for each
        """
        raise NotImplementedError(
            "TileDB does not implement similarity_search_with_score_by_index."
        )

    def similarity_search_with_score(
        self, query: str, k: int = 4, search_k: int = -1
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(embedding, k, search_k)
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, search_k: int = -1, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding, k, search_k
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_index(
        self, docstore_index: int, k: int = 4, search_k: int = -1, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to docstore_index.

        Args:
            docstore_index: Index of document in docstore
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_index(
            docstore_index, k, search_k
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self, query: str, k: int = 4, search_k: int = -1, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, search_k)
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = DEFAULT_METRIC,
        array_uri: str = "/tmp/tiledb_array",
    ) -> TileDB:
        if metric not in INDEX_METRICS:
            raise ValueError(
                (
                    f"Unsupported distance metric: {metric}. "
                    f"Expected one of {list(INDEX_METRICS)}"
                )
            )
        tiledb_vs = dependable_tiledb_import()
        if not embeddings:
            raise ValueError("embeddings must be provided to build a TileDB index")

        try:
            tiledb.group_create(array_uri)
        except tiledb.TileDBError as err:
            raise err
        group = tiledb.Group(array_uri, "w")
        vector_array_uri = f"{group.uri}/{VECTOR_ARRAY_NAME}"
        docs_uri = f"{group.uri}/{DOCUMENTS_ARRAY_NAME}"
        source_uri = f"/tmp/data-{random.randint(0, 999999)}"
        if os.path.isfile(source_uri):
            raise RuntimeError(f"Directory exists: {source_uri}")
        f = open(source_uri, "wb")
        source_type = "F32BIN"
        np.array([len(embeddings), len(embeddings[0])], dtype="uint32").tofile(f)
        np.array(embeddings).astype(np.float32).tofile(f)
        f.close()
        index = tiledb_vs.ingestion.ingest(
            index_type="IVF_FLAT",
            array_uri=vector_array_uri,
            source_uri=source_uri,
            source_type=source_type,
        )
        group.add(vector_array_uri, name=VECTOR_ARRAY_NAME)
        os.remove(source_uri)

        dim = tiledb.Dim(
            name="id",
            domain=(0, len(texts) - 1),
            dtype=np.dtype(np.int32),
        )
        dom = tiledb.Domain(dim)

        text_attr = tiledb.Attr(name="text", dtype=np.dtype("U1"), var=True)
        attrs = [text_attr]
        if metadatas is not None:
            metadata_attr = tiledb.Attr(name="metadata", dtype=np.uint8, var=True)
            attrs.append(metadata_attr)
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=attrs,
        )
        tiledb.Array.create(docs_uri, schema)
        with tiledb.open(docs_uri, "w") as A:
            data = {}
            data["text"] = np.array(texts)
            if metadatas is not None:
                metadata_attr = np.array(
                    [
                        np.frombuffer(pickle.dumps(metadata), dtype=np.uint8)
                        for metadata in metadatas
                    ],
                    dtype='O',
                )
                data["metadata"] = metadata_attr

            A[0:len(texts)] = data
        group.add(docs_uri, name=DOCUMENTS_ARRAY_NAME)
        group.close()
        return cls(embedding.embed_query, index, metric, tiledb.open(docs_uri, "r"))

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = DEFAULT_METRIC,
        array_uri: str = "/tmp/tiledb_array",
        rate_limiter: RateLimiter = RateLimiter(max_calls=100, period=60),
        **kwargs: Any,
    ) -> TileDB:
        """Construct a TileDB index from raw documents.

        Args:
            texts: List of documents to index.
            embedding: Embedding function to use.
            metadatas: List of metadata dictionaries to associate with documents.
            metric: Metric to use for indexing. Defaults to "euclidean".
            array_uri: The URI to write the TileDB arrays
            rate_limiter: RateLimiter for embeddings generation

        Example:
            .. code-block:: python

                from langchain import TileDB
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                index = TileDB.from_texts(texts, embeddings)
        """
        embeddings = []
        if rate_limiter is None:
            embeddings = embedding.embed_documents(texts)
        else:
            for i in range(len(texts)):
                with rate_limiter:
                    embeddings.append(embedding.embed_documents(texts[i])[0])
        return cls.__from(
            texts, embeddings, embedding, metadatas, metric, array_uri
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = DEFAULT_METRIC,
        array_uri: str = "/tmp/tiledb_array",
        **kwargs: Any,
    ) -> TileDB:
        """Construct TileDB index from embeddings.

        Args:
            text_embeddings: List of tuples of (text, embedding)
            embedding: Embedding function to use.
            metadatas: List of metadata dictionaries to associate with documents.
            metric: Metric to use for indexing. Defaults to "euclidean".
            array_uri: The URI to write the TileDB arrays

        Example:
            .. code-block:: python

                from langchain import TileDB
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                db = TileDB.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts, embeddings, embedding, metadatas, metric, array_uri
        )

    @classmethod
    def load(
        cls,
        array_uri: str,
        embeddings: Embeddings,
    ) -> TileDB:
        """Load a TileDB index from a URI.

        Args:
            array_uri: The URI of the TileDB array.
            embeddings: Embeddings to use when generating queries.
        """
        tiledb_vs = dependable_tiledb_import()

        group = tiledb.Group(array_uri)
        vector_array_uri = group[VECTOR_ARRAY_NAME].uri
        documents_array_uri = group[DOCUMENTS_ARRAY_NAME].uri

        index = tiledb_vs.index.IVFFlatIndex(uri=vector_array_uri, dtype=np.float32)
        docs_array = tiledb.open(uri=documents_array_uri)

        return cls(embeddings.embed_query, index, DEFAULT_METRIC, docs_array)
