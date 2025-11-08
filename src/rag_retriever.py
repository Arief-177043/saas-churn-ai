from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(notes_path="data/customer_notes.jsonl"):
    docs, ids = [], []
    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            docs.append(j["text"])
            ids.append(j["customer_id"])
    embeddings = model.encode(docs, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, "models/vectorstore.faiss")
    with open("models/docs_meta.json", "w") as f:
        json.dump({"docs": docs, "meta": ids}, f)
    print("✅ RAG index built")

def retrieve(customer_id, query, topk=3):
    index = faiss.read_index("models/vectorstore.faiss")
    with open("models/docs_meta.json") as f:
        data = json.load(f)
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb).astype("float32"), topk)
    results = [{"doc": data["docs"][i], "customer_id": data["meta"][i]} for i in I[0]]
    return [r for r in results if r["customer_id"] == customer_id]
