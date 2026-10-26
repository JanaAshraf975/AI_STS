import chromadb
from sentence_transformers import CrossEncoder
from Modules.embedding import load_model

embedding_model = load_model()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def build_vectoredb(chunks):
    client = chromadb.Client()

    try:
        client.delete_collection("myCollection")   
    except Exception:
        pass

    collection = client.get_or_create_collection(name="myCollection")
    embeddings = embedding_model.encode(chunks)
    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
    )
    return collection


def retrive_context(question, collection, top_k=3, fetch_k=10):
    query_embedding = embedding_model.encode(question)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(fetch_k, collection.count()),
    )
    docs = results["documents"][0]

    scores = reranker.predict([(question, doc) for doc in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return "\n\n".join(doc for doc, _ in ranked[:top_k])