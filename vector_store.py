import lancedb
import pyarrow as pa
from haystack import Document

class Vectorstore:
    def __init__(self, embedder, top_k=3) -> None:
        # 1. Connect to local DB file
        self.db = lancedb.connect("/tmp/data")

        # 2. Match column sizes to your explicit model dimension (384)
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
        print("data added")
        self.table.add(data)
        return "successfully added data"
    
    def search(self, query):
        # 1. Convert text to Haystack document and generate query embeddings
        embedding = self.embedder.embed_q(query)
        
        # 2. Extract embedding from the first returned query document
    

        print("proccessing with the result")
            
        # 3. Use the updated LanceDB QueryBuilder syntax to isolate vector attributes
        results = self.table.search(embedding).metric("cosine").limit(self.k).to_list()
        return results
