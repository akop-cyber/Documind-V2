import lancedb
import pyarrow as pa
from haystack import Document

class Vectorstore:
    def __init__(self, embedder, top_k) -> None:

        self.db = lancedb.connect("/tmp/data")

        schema = pa.schema([
            pa.field("docs", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 384)),
            pa.field("id", pa.int32())
        ])

        self.table = self.db.create_table("vector_store", schema=schema, mode="overwrite")
        self.k = top_k
        self.embedder = embedder

    def add_vectors(self, docs):
        data = []
        for i, chunk in enumerate(docs):
        
            vector_data = chunk.embedding
            if hasattr(vector_data, "tolist"):
                vector_data = vector_data.tolist()

            data.append({
                "docs": chunk.content,
                "vector": vector_data,
                "id": i
            })
      
        self.table.add(data)
       
    
    def search(self, query):
      
        embedding = self.embedder.embed_q(query)
        
        results = self.table.search(embedding).metric("cosine").limit(self.k).to_list()
        return results
