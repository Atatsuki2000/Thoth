from retriever import build_index, get_top_k

with open("test_corpus.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

build_index(texts)

query = "Which Python library is used for creating plots?"
docs = get_top_k(query, k=3)
print("Top 3 results:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")
