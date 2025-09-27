from rank_bm25 import BM25Okapi
from utils.populate_database import load_documents, split_documents, calculate_chunk_ids

def tokenize(corpus):
    return [doc.page_content.lower().split() for doc in corpus]

def context_bm25(query: str):
    documents = load_documents()
    chunks = split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(chunks)
    tokenized_chunks = tokenize(chunks_with_ids)
    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    # Rank top results
    top_n = bm25.get_top_n(tokenized_query, chunks_with_ids, n=5)
    # for result in top_n:
    #     print(result)
    #     print('\n')
    content = [doc.page_content for doc in top_n]
    sources = [doc.metadata.get("id", None) for doc in top_n]

    return top_n, sources, chunks_with_ids
# query = "What are the failure modes of a pump?"
# top_n, sources, chunks_with_ids = context_bm25(query)
# sources = set([doc.metadata.get("source", None) for doc in chunks_with_ids])
# print(sources)
