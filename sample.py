import re
from collections import defaultdict
from rank_bm25 import BM25Okapi

# ------------------------
# 1. Tokenization
# ------------------------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# ------------------------
# 2. Edge n-gram generator
# ------------------------
def edge_ngrams(token, min_gram=2, max_gram=10):
    return [token[:i] for i in range(min_gram, min(len(token), max_gram) + 1)]

# ------------------------
# 3. Build index
# ------------------------
class MiniSearchEngine:
    def __init__(self, docs):
        self.docs = docs
        self.tokenized_docs = [tokenize(doc) for doc in docs]

        # BM25 model (for ranking)
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Inverted index: ngram -> set(doc_ids)
        self.index = defaultdict(set)

        self._build_index()

    def _build_index(self):
        for doc_id, tokens in enumerate(self.tokenized_docs):
            for token in tokens:
                for gram in edge_ngrams(token):
                    self.index[gram].add(doc_id)

    # ------------------------
    # 4. Search
    # ------------------------
    def search(self, query, top_k=5):
        query_tokens = tokenize(query)

        # Step 1: candidate retrieval using n-grams
        candidate_docs = set()

        for token in query_tokens:
            grams = edge_ngrams(token)
            for gram in grams:
                if gram in self.index:
                    candidate_docs.update(self.index[gram])

        if not candidate_docs:
            return []

        candidate_docs = list(candidate_docs)

        # Step 2: BM25 ranking
        scores = self.bm25.get_scores(query_tokens)

        # Only rank candidates
        scored = [(doc_id, scores[doc_id]) for doc_id in candidate_docs]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(self.docs[doc_id], score) for doc_id, score in scored[:top_k]]


# ------------------------
# Example usage
# ------------------------
documents = [
    "Apple iPhone 15 Pro Max",
    "Samsung Galaxy S23 Ultra",
    "Apple MacBook Pro M3",
    "Dell XPS 13 Laptop",
    "iPhone charger fast charging cable"
]

engine = MiniSearchEngine(documents)

queries = ["iph", "apple", "laptop", "charger"]

for q in queries:
    print(f"\nQuery: {q}")
    results = engine.search(q)
    for doc, score in results:
        print(f"  {score:.2f} - {doc}")
