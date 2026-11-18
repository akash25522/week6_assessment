import os
import uuid
from contextlib import asynccontextmanager

import fitz
from docx import Document
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


COLLECTION_NAME = "week6_assessment"
API_KEY = os.getenv("API_KEY", "week6-secret-key")

model = None
client = QdrantClient(url="http://localhost:6333")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = SentenceTransformer("all-MiniLM-L6-v2")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

    yield


app = FastAPI(title="Semantic Document Search API", lifespan=lifespan)


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: str | None = None
    min_score: float | None = None


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/search")
def search_documents(
    request: SearchRequest,
    _: None = Depends(verify_api_key)
):
    vector = model.encode(request.query).tolist()

    conditions = []

    if request.doc_type:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=request.doc_type)
            )
        )

    search_filter = (
        models.Filter(must=conditions)
        if conditions else None
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=search_filter,
        limit=request.top_k,
        with_payload=True
    ).points

    response = []

    for point in results:
        if request.min_score is not None and point.score < request.min_score:
            continue

        payload = point.payload or {}

        response.append({
            "source": payload.get("source"),
            "page": payload.get("page"),
            "section": payload.get("section"),
            "score": point.score,
            "text": payload.get("text")
        })

    return {"results": response}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key)
):
    filename = file.filename
    extension = os.path.splitext(filename)[1].lower()
    content = await file.read()

    pages = []

    if extension == ".txt":
        pages.append((1, content.decode("utf-8")))

    elif extension == ".pdf":
        pdf = fitz.open(stream=content, filetype="pdf")

        for page_number, page in enumerate(pdf, start=1):
            pages.append((page_number, page.get_text()))

    elif extension == ".docx":
        import io

        document = Document(io.BytesIO(content))
        text = "\n".join(p.text for p in document.paragraphs)
        pages.append((1, text))

    else:
        raise HTTPException(
            status_code=400,
            detail="Only TXT, PDF and DOCX files are supported"
        )

    points = []

    for page_number, text in pages:
        words = text.split()

        for index in range(0, len(words), 100):
            chunk = " ".join(words[index:index + 100])

            if not chunk.strip():
                continue

            vector = model.encode(chunk).tolist()

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{filename}-{page_number}-{index}-{chunk}"
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "source": filename,
                        "page": page_number,
                        "section": filename,
                        "doc_type": extension.replace(".", "").upper(),
                        "chunk_index": index // 100,
                        "text": chunk
                    }
                )
            )

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

    return {
        "filename": filename,
        "chunks_indexed": len(points),
        "message": "Document indexed successfully"
    }