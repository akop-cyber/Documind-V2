import re
from loader import Loader
from chunker import Chunker
from embedder import Embedder
from bm25 import BM25
from haystack import Document
from vector_store import Vectorstore
import time
# 1. Load the raw document text
start = time.time()
loading_mech = Loader("portfolio.pdf")
docs = loading_mech.load()
print(docs)
print(f"loaded the pdf:{time.time() - start:.2f}s")
#print(docs[0].content)

#print("docs done")

start  = time.time()
chunking_mech = Chunker(docs)
chunks = chunking_mech.chunk()
print(f"chunked the pdf: {time.time() - start:.2f}s")

#print("chunking done")

start = time.time()
embedder = Embedder()
embeddings = embedder.embed(chunks)
print(f"generated the embeddings {time.time() - start:.2f}s")

start = time.time()
bm25 = BM25()
data = bm25.add(chunks)
print(f"added data to bm25: {time.time() - start:.2f}s")

#print("added the chunks in bm25")
#query = "who is aarav"
#doc = Document(content = query)
#result = bm25.search(doc)
#print(result)

start = time.time()
store = Vectorstore(embedder=embedder)
store.add_vectors(embeddings)
print(f"added data to vector store {time.time() - start:.2f}s")

#print(f"\n=== TABLE STATUS ===")
#print(f"Rows in table: {store.table.count_rows()}")
#print(f"First row sample: {store.table.head()}")

start = time.time()
result = store.search("Who is aarav")

print(f"after searhing in the vector store {time.time() - start:.2f}s")