from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack import Document

class BM25:
    def __init__(self,top_k = 20) -> None:
        self.store = InMemoryDocumentStore()
        self.model = InMemoryBM25Retriever(
            document_store=self.store,
            top_k=top_k
        )


    def add(self,docs):
        self.store.write_documents(docs)

    def search(self,query):
        doc = Document(content = query)

        results = self.model.run(query=doc.content)

        return results["documents"]
    