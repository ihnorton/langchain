"""Wrapper around TileDB vector database."""
from __future__ import annotations

import pickle
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tiledb

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

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

    def process_index_results(
        self,
        ids: List[int],
        scores: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Turns TileDB results into a list of documents and scores.

        Args:
            ids: List of indices of the documents in the index.
            scores: List of distances of the documents in the index.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs
        Returns:
            List of Documents and scores.
        """
        docs = []
        docs_array = tiledb.open(self.docs_array_uri, "r")
        for idx, score in zip(ids, scores):
            if idx == 0 and score == 0:
                continue
            if idx == MAX_UINT64 and score == MAX_FLOAT_32:
                continue
            doc = docs_array[idx]
            if doc is None or len(doc["text"]) == 0:
                raise ValueError(f"Could not find document for id {idx}, got {doc}")
            pickled_metadata = doc.get("metadata")
            result_doc = Document(page_content=str(doc["text"][0]))
            if pickled_metadata is not None:
                metadata = pickle.loads(
                    np.array(pickled_metadata.tolist()).astype(np.uint8).tobytes()
                )
                result_doc.metadata = metadata
            if filter is not None:
                filter = {
                    key: [value] if not isinstance(value, list) else value
                    for key, value in filter.items()
                }
                if all(
                    result_doc.metadata.get(key) in value
                    for key, value in filter.items()
                ):
                    docs.append((result_doc, score))
            else:
                docs.append((result_doc, score))
        docs_array.close()
        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            docs = [(doc, score) for doc, score in docs if score <= score_threshold]
        return docs[:k]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        d, i = self.index.query(
            np.array([np.array(embedding).astype(np.float32)]).astype(np.float32),
            k=k if filter is None else fetch_k,
        )
        return self.process_index_results(
            ids=i[0], scores=d[0], filter=filter, k=k, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of documents most similar to the query text with
            L2 distance in float. Lower score represents more similarity.
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        scores, indices = self.index.query(
            np.array([np.array(embedding).astype(np.float32)]).astype(np.float32),
            k=fetch_k if filter is None else fetch_k * 2,
        )
        results = self.process_index_results(
            ids=indices[0],
            scores=scores[0],
            filter=filter,
            k=fetch_k if filter is None else fetch_k * 2,
        )
        embeddings = [
            self.embedding.embed_documents([doc.page_content])[0] for doc, _ in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        docs_and_scores = []
        for i in mmr_selected:
            docs_and_scores.append(results[i])
        return docs_and_scores

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering (if needed) to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

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
            ids = [random.randint(0, MAX_UINT64 - 1) for _ in texts]
        external_ids = np.array(ids).astype(np.uint64)

        input_vectors = np.array(embeddings).astype(np.float32)
        index = tiledb_vs.ingestion.ingest(
            index_type="IVF_FLAT",
            index_uri=vector_array_uri,
            input_vectors=input_vectors,
            external_ids=external_ids,
            partitions=1,
        )
        group.add(vector_array_uri, name=VECTOR_ARRAY_NAME)

        dim = tiledb.Dim(
            name="id",
            domain=(0, MAX_UINT64 - 1),
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
                    metadata_attr[i] = np.frombuffer(
                        pickle.dumps(metadata), dtype=np.uint8
                    )
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
        embeddings = self.embedding.embed_documents(texts)
        if ids is None:
            ids = [random.randint(0, 100000) for _ in texts]

        external_ids = np.array(ids).astype(np.uint64)
        vectors = np.empty((len(embeddings)), dtype="O")
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
        **kwargs: Any,
    ) -> TileDB:
        """Construct a TileDB index from raw documents.

        Args:
            texts: List of documents to index.
            embedding: Embedding function to use.
            metadatas: List of metadata dictionaries to associate with documents.
            metric: Metric to use for indexing. Defaults to "euclidean".
            array_uri: The URI to write the TileDB arrays

        Example:
            .. code-block:: python

                from langchain import TileDB
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                index = TileDB.from_texts(texts, embeddings)
        """
        embeddings = []
        embeddings = embedding.embed_documents(texts)
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
