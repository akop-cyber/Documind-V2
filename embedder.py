from sentence_transformers import SentenceTransformer
from haystack import Document

class Embedder:
    def __init__(self):
        # Load once, reuse forever
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed(self, documents: list[Document]) -> list[Document]:
        # Extract texts
        texts = [doc.content for doc in documents]
        
        # Batch embed (super fast!)
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=2  # Tune this!
        )
        
        # Attach embeddings back to documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb.tolist()
        
        return documents
    
    def embed_q(self, query: str) -> list[float]:
        """For single query embedding (used in Vectorstore.search)"""
        return self.model.encode(query, convert_to_numpy=True).tolist()