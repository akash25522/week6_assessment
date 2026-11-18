import os
import time
import uuid
import numpy as np
import fitz
import faiss

from docx import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# ANN AND HNSW EXPLANATION
print("ANN AND HNSW")
print("1. ANN means Approximate Nearest Neighbor search.")
print("2. ANN finds similar vectors faster than exact search.")
print("3. HNSW uses a hierarchical graph structure.")
print("4. HNSW improves search speed using graph navigation.")
print("5. HNSW provides a balance between speed and accuracy.")


# FAISS EXPERIMENT
print("\nFAISS EXPERIMENT")

np.random.seed(42)

vectors = np.random.random((50000, 128)).astype("float32")
query = np.random.random((1, 128)).astype("float32")

faiss.normalize_L2(vectors)
faiss.normalize_L2(query)

# Exact Search
exact_index = faiss.IndexFlatIP(128)
exact_index.add(vectors)

start = time.time()
exact_scores, exact_ids = exact_index.search(query, 10)
exact_time = time.time() - start

# HNSW Search
hnsw_index = faiss.IndexHNSWFlat(
    128,
    32,
    faiss.METRIC_INNER_PRODUCT
)

hnsw_index.hnsw.efConstruction = 100
hnsw_index.hnsw.efSearch = 64
hnsw_index.add(vectors)

start = time.time()
hnsw_scores, hnsw_ids = hnsw_index.search(query, 10)
hnsw_time = time.time() - start

recall = len(
    set(exact_ids[0]) & set(hnsw_ids[0])
) / 10

print("Exact Search:", round(exact_time * 1000, 2), "ms")
print("HNSW Search:", round(hnsw_time * 1000, 2), "ms")
print("Recall@10:", round(recall, 2))


# QDRANT STORAGE
print("\nQDRANT VECTOR STORAGE")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(url="http://localhost:6333")

collection_name = "week6_assessment"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=100
    )
)


# DOCUMENT EXTRACTION
def extract_document(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        pdf = fitz.open(path)
        return [
            (page.get_text(), i + 1)
            for i, page in enumerate(pdf)
        ]

    elif extension == ".docx":
        doc = Document(path)
        return [
            ("\n".join(p.text for p in doc.paragraphs), 1)
        ]

    else:
        with open(path, "r", encoding="utf-8") as file:
            return [(file.read(), 1)]


# CHUNK DOCUMENTS
points = []

for filename in os.listdir("documents"):

    path = os.path.join("documents", filename)

    if not os.path.isfile(path):
        continue

    pages = extract_document(path)

    for text, page in pages:

        words = text.split()

        chunks = [
            " ".join(words[i:i + 100])
            for i in range(0, len(words), 100)
        ]

        for index, chunk in enumerate(chunks):

            embedding = model.encode(chunk).tolist()

            unique_value = f"{filename}_{page}_{index}_{chunk}"

            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, unique_value)
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "source": filename,
                        "page": page,
                        "section": filename.split(".")[0],
                        "doc_type": os.path.splitext(filename)[1][1:].upper(),
                        "chunk_index": index,
                        "text": chunk
                    }
                )
            )


# UPSERT
client.upsert(
    collection_name=collection_name,
    points=points,
    wait=True
)

time.sleep(1)

info = client.get_collection(collection_name)

print("Total Points:", info.points_count)
print("Indexed Vectors:", info.indexed_vectors_count)
print("Optimizer Status:", info.status)


# PAYLOAD INDEXES
client.create_payload_index(
    collection_name=collection_name,
    field_name="doc_type",
    field_schema=models.PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="page",
    field_schema=models.PayloadSchemaType.INTEGER
)

print("\nPayload Indexes Created Successfully")


# FILTERED SEARCH
print("\nFILTERED SEARCH")

query_vector = model.encode(
    "company employee policies"
).tolist()

results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value="TXT")
            ),
            models.FieldCondition(
                key="page",
                range=models.Range(gte=3)
            )
        ]
    ),
    limit=5
).points

print("Filtered Results Count:", len(results))

for result in results:
    print(
        result.payload["source"],
        result.payload["page"],
        result.score
    )


# RE-UPSERT TEST
print("\nRE-UPSERT TEST")

client.upsert(
    collection_name=collection_name,
    points=points,
    wait=True
)

time.sleep(1)

print(
    "Points after re-upsert:",
    client.get_collection(collection_name).points_count
)


# DELETE ONE SOURCE
print("\nDELETE SOURCE")

if points:
    source_to_delete = points[0].payload["source"]

    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(
                            value=source_to_delete
                        )
                    )
                ]
            )
        ),
        wait=True
    )

    print("Deleted Source:", source_to_delete)

print(
    "Remaining Points:",
    client.get_collection(collection_name).points_count
)


# PGVECTOR VS QDRANT
print("\nPGVECTOR VS QDRANT")
print("-" * 65)
print(f"{'Factor':<20}{'pgvector':<20}{'Qdrant':<20}")
print("-" * 65)
print(f"{'Setup':<20}{'PostgreSQL':<20}{'Standalone DB':<20}")
print(f"{'Filtering':<20}{'SQL Queries':<20}{'Payload Filters':<20}")
print(f"{'Scale':<20}{'Database Scale':<20}{'Vector Scale':<20}")
print(f"{'Choose':<20}{'SQL Integration':<20}{'Vector Search':<20}")
print("-" * 65)

print("\nTASK 3 COMPLETED")
