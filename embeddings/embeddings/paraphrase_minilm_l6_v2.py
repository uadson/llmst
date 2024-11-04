from sentence_transformers import SentenceTransformer
from query_api import questions

model = SentenceTransformer("all-MiniLM-L6-v2")

questions = questions
query = ["Quero pagar meu iptu"]

q = model.encode(query)

embeddings = model.encode(questions)

print(embeddings.shape)
print(q.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
