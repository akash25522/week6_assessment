import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans

load_dotenv()

sentences = [
    "Doctors treat sick people in hospitals.",
    "Physicians care for ill patients in clinics.",
    "Python is a popular programming language.",
    "Python is a widely used programming language.",
    "The weather is sunny today.",
    "It is a bright and warm day.",
    "Machine learning helps computers learn.",
    "Artificial intelligence makes machines smart.",
    "I like playing cricket.",
    "The train arrived at the station."
]

# Local Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

start = time.time()
local_embeddings = model.encode(sentences)
local_time = time.time() - start

print("\nLOCAL EMBEDDINGS")
print("Dimension:", local_embeddings.shape)
print("Time:", round(local_time, 4), "seconds")

# Gemini API Embeddings
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

start = time.time()

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences
)

api_embeddings = np.array(
    [embedding.values for embedding in response.embeddings]
)

api_time = time.time() - start

print("\nGEMINI API EMBEDDINGS")
print("Dimension:", api_embeddings.shape)
print("Time:", round(api_time, 4), "seconds")

print("\nTIME COMPARISON")
print("Local Model:", round(local_time, 4), "seconds")
print("Gemini API:", round(api_time, 4), "seconds")

# Cosine Similarity From Scratch
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


print("\nCOSINE SIMILARITY TESTS")

print("[1,2,3] vs [2,4,6] ->",
      round(cosine_similarity([1, 2, 3], [2, 4, 6]), 2))

print("[1,0] vs [0,1] ->",
      round(cosine_similarity([1, 0], [0, 1]), 2))

print("[1,0] vs [-1,0] ->",
      round(cosine_similarity([1, 0], [-1, 0]), 2))

# Scikit-learn Comparison
print("\nSCIKIT-LEARN COMPARISON")

print(sklearn_cosine([[1, 2, 3]], [[2, 4, 6]])[0][0])

# Similarity Matrix
similarity_matrix = sklearn_cosine(local_embeddings)

print("\nSIMILARITY MATRIX")
print(np.round(similarity_matrix, 2))

# Top 3 Similar Pairs
pairs = []

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        pairs.append((similarity_matrix[i][j], i, j))

pairs.sort(reverse=True)

print("\nTOP 3 SIMILAR PAIRS")

for score, i, j in pairs[:3]:
    print(f"{sentences[i]} <-> {sentences[j]} : {score:.4f}")

# No Common Words Example
print("\nNO COMMON WORDS EXAMPLE")

print(sentences[0])
print(sentences[1])

print("Similarity:",
      round(similarity_matrix[0][1], 4))

# Semantic Search
queries = [
    "A small animal is resting.",
    "Programming languages",
    "Artificial intelligence"
]

print("\nSEMANTIC SEARCH")

for query in queries:
    query_embedding = model.encode([query])

    scores = sklearn_cosine(
        query_embedding, local_embeddings
    )[0]

    best_index = np.argmax(scores)

    print("\nQuery:", query)
    print("Result:", sentences[best_index])
    print("Score:", round(scores[best_index], 4))

# Duplicate Detection
print("\nDUPLICATE DETECTION")

threshold = 0.75

for score, i, j in pairs:
    if score >= threshold:
        print(
            f"Duplicate: {sentences[i]} | "
            f"{sentences[j]} | {score:.4f}"
        )

# K-Means Clustering
print("\nK-MEANS CLUSTERING")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(local_embeddings)

cluster_names = {
    0: "Technology",
    1: "Daily Life",
    2: "General Topics"
}

for i, label in enumerate(labels):
    print(
        f"{cluster_names[label]}: {sentences[i]}"
    )

# Local vs API Comparison
print("\nLOCAL VS API EMBEDDING MODEL")

print("1. Local models can run without internet after downloading.")
print("2. API models require internet and an API key.")
print("3. Local models reduce API usage costs.")
print("4. API models are useful for scalable applications.")
