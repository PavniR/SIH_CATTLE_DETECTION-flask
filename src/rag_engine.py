
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pinecone import Pinecone, ServerlessSpec
import os

# Embedding + LLM
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
gen_model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

# Pinecone init
os.environ["PINECONE_API_KEY"] = "pcsk_6q1DjW_Ew7zauktgZ3qiykoEAikosuzKsgL428au2XxjiG32uQyugBMTQR1epyTmbQwdyp"
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "cattle-emb-384"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

def add_knowledge(texts):
    embeddings = embed_model.encode(texts)
    vectors = [
        (f"id-{i}", embeddings[i].tolist(), {"text": texts[i]})
        for i in range(len(texts))
    ]
    index.upsert(vectors=vectors)

def answer_question(question, top_k=2, max_tokens=100):
    q_emb = embed_model.encode([question])[0]
    res = index.query(vector=q_emb.tolist(), top_k=top_k, include_metadata=True)
    context = " ".join([m['metadata']['text'] for m in res['matches']])
    prompt = f"Answer the question based on the context below:\nContext: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = gen_model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

