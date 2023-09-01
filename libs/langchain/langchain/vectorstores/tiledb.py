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
MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
MAX_FLOAT_32 = np.finfo(np.dtype("float32")).max


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
            db = TileDB(embedding_function, index, metric, docs_array_uri)

    """

    def __init__(
        self,
        embedding: Embeddings,
        index: Any,
        metric: str,
        docs_array_uri: str,
    ):
        """Initialize with necessary components."""
        self.embedding = embedding
        self.embedding_function = embedding.embed_query
        self.index = index
        self.metric = metric
        self.docs_array_uri = docs_array_uri

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Accept embedding object directly
        return None

    def process_index_results(self, idxs: List[int], dists: List[float]) -> List[Tuple[Document, float]]:
        """Turns TileDB results into a list of documents and scores.

        Args:
            idxs: List of indices of the documents in the index.
            dists: List of distances of the documents in the index.
        Returns:
            List of Documents and scores.
        """
        docs = []
        docs_array = tiledb.open(self.docs_array_uri, "r")
        for idx, dist in zip(idxs, dists):
            if idx == 0 and dist == 0:
                continue
            if idx == MAX_UINT64 and dist == MAX_FLOAT_32:
                continue
            doc = docs_array[idx]
            if doc is None or len(doc["text"]) == 0:
                raise ValueError(f"Could not find document for id {idx}, got {doc}")
            pickled_metadata = doc.get("metadata")
            if pickled_metadata is not None:
                metadata = pickle.loads(
                    np.array(pickled_metadata.tolist()).astype(np.uint8).tobytes()
                )
                docs.append(
                    (Document(page_content=str(doc["text"][0]), metadata=metadata), dist)
                )
            else:
                docs.append(
                    (Document(page_content=str(doc["text"][0])), dist)
                )
        docs_array.close()
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
        d, i = self.index.query(
            np.array([np.array(embedding).astype(np.float32)]).astype(np.float32), k=k
        )
        return self.process_index_results(idxs=i[0], dists=d[0])

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
        ids: Optional[List[str]] = None,
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
        if ids is None:
            ids = [random.randint(0, 100000) for _ in texts]
        external_ids = np.array(ids).astype(np.uint64)

        input_vectors = np.array(embeddings).astype(np.float32)
        index = tiledb_vs.ingestion.ingest(
            index_type="IVF_FLAT",
            index_uri=vector_array_uri,
            input_vectors=input_vectors,
            external_ids=external_ids,
            partitions=1
        )
        group.add(vector_array_uri, name=VECTOR_ARRAY_NAME)

        dim = tiledb.Dim(
            name="id",
            domain=(0, MAX_UINT64-1),
            dtype=np.dtype(np.uint64),
        )
        dom = tiledb.Domain(dim)

        text_attr = tiledb.Attr(name="text", dtype=np.dtype("U1"), var=True)
        attrs = [text_attr]
        if metadatas is not None:
            metadata_attr = tiledb.Attr(name="metadata", dtype=np.uint8, var=True)
            attrs.append(metadata_attr)
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            allows_duplicates=False,
            attrs=attrs,
        )
        tiledb.Array.create(docs_uri, schema)
        with tiledb.open(docs_uri, "w") as A:
            if external_ids is None:
                external_ids = np.zeros(len(texts), dtype=np.uint64)
                for i in range(len(texts)):
                    external_ids[i] = i
            data = {}
            data["text"] = np.array(texts)
            if metadatas is not None:
                metadata_attr = np.empty([len(metadatas)], dtype=object)
                i = 0
                for metadata in metadatas:
                    metadata_attr[i] = np.frombuffer(pickle.dumps(metadata), dtype=np.uint8)
                    i += 1
                data["metadata"] = metadata_attr

            A[external_ids] = data
        group.add(docs_uri, name=DOCUMENTS_ARRAY_NAME)
        group.close()
        return cls(embedding, index, metric, docs_uri)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        external_ids = np.array(ids).astype(np.uint64)
        self.index.delete_batch(external_ids=external_ids)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        rate_limiter: RateLimiter = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = []
        if rate_limiter is None:
            embeddings = self.embedding.embed_documents(texts)
        else:
            for i in range(len(texts)):
                with rate_limiter:
                    embeddings.append(self.embedding.embed_documents(texts[i])[0])
        if ids is None:
            ids = [random.randint(0, 100000) for _ in texts]

        external_ids = np.array(ids).astype(np.uint64)
        vectors = np.empty((len(embeddings)), dtype='O')
        for i in range(len(embeddings)):
            vectors[i] = np.array(embeddings[i], dtype=np.float32)
        self.index.update_batch(vectors=vectors, external_ids=external_ids)

        docs = {}
        docs["text"] = np.array(texts)
        if metadatas is not None:
            metadata_attr = np.empty([len(metadatas)], dtype=object)
            i = 0
            for metadata in metadatas:
                metadata_attr[i] = np.frombuffer(pickle.dumps(metadata), dtype=np.uint8)
                i += 1
            docs["metadata"] = metadata_attr

        docs_array = tiledb.open(self.docs_array_uri, "w")
        docs_array[external_ids] = docs
        docs_array.close()
        return ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        array_uri: str = "/tmp/tiledb_array",
        rate_limiter: RateLimiter = None,
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
            texts, embeddings, embedding, metadatas, ids, metric, array_uri
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
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
            texts, embeddings, embedding, metadatas, ids, metric, array_uri
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

        index = tiledb_vs.ivf_flat_index.IVFFlatIndex(uri=vector_array_uri)

        return cls(embeddings, index, DEFAULT_METRIC, documents_array_uri)

    def consolidate_updates(self):
        self.index = self.index.consolidate_updates()
