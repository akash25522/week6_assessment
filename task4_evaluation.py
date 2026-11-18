from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "week6_assessment"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")

evaluation_queries = [
    ("What are company working hours?", "hr_policy.txt", 1),
    ("How often are performance reviews conducted?", "hr_policy.txt", 1),
    ("How should employees maintain professional behavior?", "hr_policy.txt", 1),
    ("How many casual leaves are available?", "leave_policy.txt", 1),
    ("When is medical proof required?", "leave_policy.txt", 1),
    ("Who should receive leave requests?", "leave_policy.txt", 1),
    ("Why are strong passwords important?", "it_policy.txt", 1),
    ("Who approves software installation?", "it_policy.txt", 1),
    ("How should company data be protected?", "it_policy.txt", 1),
    ("How should confidential information be protected?", "security_manual.txt", 1),
    ("Should passwords be shared?", "security_manual.txt", 1),
    ("What should employees do after a security incident?", "security_manual.txt", 1),
    ("What is multi-factor authentication?", "security_manual.txt", 1),
    ("When should computers be locked?", "security_manual.txt", 1),
    ("How can employees protect company accounts?", "security_manual.txt", 1),
]


def evaluate(hnsw_ef):
    hit_counts = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []

    for query, expected_source, expected_page in evaluation_queries:
        vector = model.encode(query).tolist()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=5,
            search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            with_payload=True
        ).points

        rank = None

        for index, result in enumerate(results, start=1):
            payload = result.payload or {}

            if (
                payload.get("source") == expected_source
                and payload.get("page") == expected_page
            ):
                rank = index
                break

        for k in [1, 3, 5]:
            if rank is not None and rank <= k:
                hit_counts[k] += 1

        reciprocal_ranks.append(
            1 / rank if rank is not None else 0
        )

    total = len(evaluation_queries)

    return {
        "Hit@1": hit_counts[1] / total,
        "Hit@3": hit_counts[3] / total,
        "Hit@5": hit_counts[5] / total,
        "MRR": sum(reciprocal_ranks) / total
    }


print("RETRIEVAL EVALUATION")
print("Total Queries:", len(evaluation_queries))

config_a = evaluate(32)
config_b = evaluate(128)

print("\nCONFIGURATION COMPARISON")
print("-" * 65)
print(f"{'Metric':<15}{'Config A (32)':<20}{'Config B (128)':<20}")
print("-" * 65)

for metric in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:
    print(
        f"{metric:<15}"
        f"{config_a[metric]:<20.2f}"
        f"{config_b[metric]:<20.2f}"
    )

print("-" * 65)