"""RAG (Retrieval-Augmented Generation) extension.

Provides an abstract :class:`RAGExtension` that users subclass to plug in
their own retrieval backend (ChromaDB, LightRAG, FAISS, etc.).

Example::

    from supervisors.ext.rag import RAGExtension

    class ChromaRAG(RAGExtension):
        def __init__(self, collection):
            super().__init__()
            self._col = collection

        def retrieve(self, query, top_k=5):
            results = self._col.query(query_texts=[query], n_results=top_k)
            return results["documents"][0]

        def add_documents(self, docs, **kwargs):
            ids = [str(i) for i in range(len(docs))]
            self._col.add(documents=docs, ids=ids)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from supervisors.ext import Extension

if TYPE_CHECKING:
    from supervisors._core import Message
    from supervisors.agent import Agent


class RAGExtension(Extension):
    """Base class for RAG extensions.

    Subclass and implement :meth:`retrieve` and :meth:`add_documents` to
    connect your own retrieval backend.  The extension automatically
    enriches incoming messages by appending retrieved context to the
    message content when :attr:`auto_retrieve` is ``True``.

    Parameters:
        auto_retrieve: When ``True`` (default), every incoming message is
            enriched with retrieved context automatically.
        top_k: Number of documents to retrieve per query.
    """

    name: str = "rag"

    def __init__(self, *, auto_retrieve: bool = True, top_k: int = 5) -> None:
        self.auto_retrieve = auto_retrieve
        self.top_k = top_k

    # -- abstract interface (override these) ---------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve relevant documents for *query*.

        Returns a list of document strings.  Override in your subclass.

        Parameters:
            query: The search query.
            top_k: Maximum number of results (defaults to ``self.top_k``).
        """
        raise NotImplementedError("Subclass RAGExtension and implement retrieve()")

    def add_documents(self, docs: List[str], **kwargs: Any) -> None:
        """Ingest *docs* into the retrieval store.

        Override in your subclass.
        """
        raise NotImplementedError("Subclass RAGExtension and implement add_documents()")

    # -- lifecycle hook -------------------------------------------------------

    def on_message(self, agent: "Agent", msg: "Message") -> Optional["Message"]:
        """Enrich the message with retrieved context when auto_retrieve is on."""
        if not self.auto_retrieve:
            return None

        from supervisors._core import Message as Msg

        docs = self.retrieve(msg.content, self.top_k)
        if docs:
            context = "\n---\n".join(docs)
            enriched_content = f"{msg.content}\n\n[RAG context]\n{context}"
            return Msg(msg.sender, msg.recipient, enriched_content)
        return None


__all__ = ["RAGExtension"]
