import os
import uuid
import fitz
import numpy as np

from datetime import date
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENT_FOLDER = "documents"

model = SentenceTransformer("all-MiniLM-L6-v2")


# Extract Documents
def extract_text(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        pdf = fitz.open(file_path)
        return [
            (page.get_text(), i + 1)
            for i, page in enumerate(pdf)
        ]

    elif extension == ".docx":
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return [(text, 1)]

    else:
        with open(file_path, "r", encoding="utf-8") as file:
            return [(file.read(), 1)]


documents = []

for filename in os.listdir(DOCUMENT_FOLDER):
    path = os.path.join(DOCUMENT_FOLDER, filename)

    if os.path.isfile(path):
        for text, page in extract_text(path):
            documents.append({
                "source": filename,
                "text": text,
                "page": page,
                "doc_type": os.path.splitext(filename)[1][1:].upper()
            })


# Fixed Size Chunking
def fixed_chunk(text, size=512, overlap=50):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap

    return chunks


# Sentence Chunking
def sentence_chunk(text):
    sentences = text.replace("\n", " ").split(".")
    return [
        sentence.strip() + "."
        for sentence in sentences
        if sentence.strip()
    ]


# Recursive Chunking
def recursive_chunk(text, size=512):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=50
    )

    return splitter.split_text(text)


# Stable IDs
def stable_id(source, page, index, content):
    value = f"{source}_{page}_{index}_{content}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


# Metadata
def describe_chunk(metadata):
    return (
        f"Source: {metadata['source']}, "
        f"Page: {metadata['page']}, "
        f"Section: {metadata['section_heading']}, "
        f"Chunk: {metadata['chunk_index']}"
    )


# Chunk Comparison
all_chunks = []

for document in documents:
    text = document["text"]

    fixed = fixed_chunk(text)
    sentence = sentence_chunk(text)
    recursive = recursive_chunk(text)

    print("\nDocument:", document["source"])
    print("Fixed 512/50:", len(fixed))
    print("Sentence chunks:", len(sentence))
    print("Recursive 512/50:", len(recursive))

    for index, chunk in enumerate(recursive):
        metadata = {
            "source": document["source"],
            "page": document["page"],
            "section_heading": document["source"].split(".")[0],
            "created_date": str(date.today()),
            "chunk_index": index,
            "doc_type": document["doc_type"]
        }

        all_chunks.append({
            "id": stable_id(
                document["source"],
                document["page"],
                index,
                chunk
            ),
            "text": chunk,
            "metadata": metadata
        })


print("\nTOTAL DOCUMENTS:", len(documents))
print("TOTAL CHUNKS:", len(all_chunks))

if all_chunks:
    print("\nCHUNK METADATA:")
    print(describe_chunk(all_chunks[0]["metadata"]))


# Stable ID Test
first_run = {
    chunk["id"]: chunk["text"]
    for chunk in all_chunks
}

second_run = {
    chunk["id"]: chunk["text"]
    for chunk in all_chunks
}

print("\nSTABLE ID TEST")
print("Run 1:", len(first_run), "points")
print("Run 2:", len(second_run), "points")

assert first_run == second_run

print("No duplicate points created.")


# Edit Demonstration
if all_chunks:
    original = all_chunks[0]["text"]
    modified = original + " Updated company policy information."

    old_id = stable_id(
        all_chunks[0]["metadata"]["source"],
        all_chunks[0]["metadata"]["page"],
        all_chunks[0]["metadata"]["chunk_index"],
        original
    )

    new_id = stable_id(
        all_chunks[0]["metadata"]["source"],
        all_chunks[0]["metadata"]["page"],
        all_chunks[0]["metadata"]["chunk_index"],
        modified
    )

    print("\nEDIT TEST")
    print("Original ID:", old_id)
    print("Modified ID:", new_id)
    print("Changed:", old_id != new_id)


# Retrieval Evaluation
evaluation_queries = [
    ("What are company working hours?", "hr_policy.txt", 1),
    ("How many casual leaves are available?", "leave_policy.txt", 1),
    ("What is the sick leave requirement?", "leave_policy.txt", 1),
    ("How should employees protect passwords?", "it_policy.txt", 1),
    ("Who approves software installation?", "it_policy.txt", 1),
    ("How is confidential information protected?", "security_manual.txt", 1),
    ("What should employees do after a security incident?", "security_manual.txt", 1),
    ("What happens during employee onboarding?", "employee_manual.txt", 1),
    ("How often are team meetings conducted?", "employee_manual.txt", 1),
    ("When are performance reviews conducted?", "hr_policy.txt", 1)
]


# Chunk Size Experiment
chunk_sizes = [128, 256, 512, 1024]

print("\nCHUNK SIZE HIT-RATE EXPERIMENT")
print("-" * 55)
print(f"{'Chunk Size':<15}{'Hits':<15}{'Hit Rate':<15}")
print("-" * 55)

for size in chunk_sizes:

    experiment_chunks = []

    for document in documents:
        chunks = recursive_chunk(document["text"], size)

        for chunk in chunks:
            experiment_chunks.append({
                "text": chunk,
                "source": document["source"],
                "page": document["page"]
            })

    if not experiment_chunks:
        continue

    texts = [chunk["text"] for chunk in experiment_chunks]

    embeddings = model.encode(texts)

    hits = 0

    for query, expected_source, expected_page in evaluation_queries:

        query_embedding = model.encode([query])

        scores = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        best_index = np.argmax(scores)

        result = experiment_chunks[best_index]

        if (
            result["source"] == expected_source
            and result["page"] == expected_page
        ):
            hits += 1

    hit_rate = hits / len(evaluation_queries)

    print(
        f"{size:<15}{hits:<15}"
        f"{hit_rate:.2f}"
    )

print("-" * 55)
print("Evaluation Queries:", len(evaluation_queries))
print("Chunk Size Experiment Completed.")

