"""
Task 1 -- Permission-aware retrieval.

Chunk documents/*.txt, embed each chunk with a REAL embedding model
(your choice -- sentence-transformers, an API, etc. -- no keyword/TF-IDF
substitutes), attach ACL metadata from documents/manifest.json, and
implement retrieve() below.

Hard rule: an unauthorized chunk must never be constructible from the
return value of retrieve(). Filtering happens INSIDE this function,
before anything is returned -- never filter after the fact.
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from openai import AsyncOpenAI
from .tools import tools
import numpy as np

client = AsyncOpenAI(base_url="http://host.docker.internal:11434/v1", api_key="ollama")

# acl = json.load('documents/manifest.json')
acl = open('documents/manifest.json')
acl_json = json.loads(acl.read())


@dataclass
class Chunk:
    doc_id: str
    text: str
    acl: List[str]
    embedding: List[float] = field(default_factory=list)
    score: float = 0.0


async def load_documents(
    documents_dir: str = "documents",
) -> List[Chunk]:
    directory = Path(documents_dir)
    files = sorted(directory.glob("*.txt"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", " "],
    )

    chunks = []

    for doc_id, file in enumerate(files):
        acl_file = acl_json[file.name]

        text = file.read_text(encoding="utf-8")
        texts = splitter.split_text(text)

        for chunk_text in texts:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    text=chunk_text,
                    embedding=[], 
                    acl=acl_file["acl"],
                )
            )

    if not chunks:
        return []

    response = await client.embeddings.create(
        model="nomic-embed-text",
        input=[chunk.text for chunk in chunks],
    )

    for item in response.data:
        chunks[item.index].embedding = item.embedding

    return chunks


def is_authorized(chunk_acl: List[str], user_groups: List[str]) -> bool:
    """A chunk is authorized for a user if 'all' is in its acl, or if any
    of the user's groups intersects the chunk's acl.
    """
    return 'all' in chunk_acl or any(group in chunk_acl for group in user_groups)


_chunks_cache = None

async def retrieve(query: str, user_groups: List[str], top_k: int = 3) -> List[Chunk]:
    """Return only chunks the given user_groups are authorized to see,
    ranked by embedding similarity to `query`.
    """
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = await load_documents("documents")

    query_embeddings = await client.embeddings.create(
        model="nomic-embed-text",
        input=query,
    )
    
    query_vec = np.array(query_embeddings.data[0].embedding)
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    scored_chunks = []
    for chunk in _chunks_cache:
        if not is_authorized(chunk.acl, user_groups):
            continue

        chunk_vec = np.array(chunk.embedding)
        chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)

        sim = float(query_vec @ chunk_vec)  # cosine similarity
        if sim < 0.55:
            continue
        chunk.score = sim
        scored_chunks.append(chunk)

    scored_chunks.sort(key=lambda x: x.score, reverse=True)
    return [chunk for chunk in scored_chunks[:top_k]]
