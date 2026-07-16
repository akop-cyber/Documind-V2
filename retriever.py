class Retriever:
    def __init__(self,vector_store, bm25, top_k = 2,rrf = 60) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25
        self.top_k = top_k
        self.rrf = rrf

        
    def retrieve(self, query):
        sematic = self.vector_store.search(query)
        lexical = self.bm25.search(query)

        scores = {}
    
        for idx , chunk in enumerate(sematic):
            rank = idx + 1
            text = chunk["docs"]
            rrf_score = 1/(self.rrf + rank)

            scores[text] = scores.get(text , 0) + rrf_score

        for idx , chunk in enumerate(lexical):
            rank = idx + 1
            text = chunk.content
            rrf_score = 1/(self.rrf + rank)

            scores[text] = scores.get(text , 0) + rrf_score

        sorted_dict = dict(sorted(scores.items(),key=lambda x: x[1],reverse= True))

        retrieved_results = dict(list(sorted_dict.items())[:self.top_k])

        return retrieved_results