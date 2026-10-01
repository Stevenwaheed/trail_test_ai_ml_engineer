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
from sentence_transformers import SentenceTransformer
import json


embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


acl = json.load('documents/manifest.json')
acl = open('documents/manifest.json')
acl_json = json.loads(acl.read())


@dataclass
class Chunk:
    doc_id: str
    text: str
    acl: List[str]
    embedding: List[float] = field(default_factory=list)
    score: float = 0.0


def load_documents(documents_dir: str = "documents") -> List[Chunk]:
    """Load and chunk all documents, attaching ACL metadata from manifest.json.

    Embeddings should be computed here (or lazily) using a real embedding
    model. Paragraph-level chunking is fine given how short these documents
    are.
    """

    directory = Path(documents_dir)

    files = list(directory.glob('*.txt'))

    chuncks = []

    for id, file in enumerate(files):

        acl_file = acl_json[file.name]
        with open(file, 'r') as f:
            text = f.read()

            splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunck_overlap=5)
            texts = splitter.split_text(text)

            document_embeddings = embedding_model.encode(texts)

            for text, embedding in zip(texts, document_embeddings):

                chuncks.append(
                    Chunk(
                    doc_id=id,
                    text=texts,
                    embedding=embedding,
                    acl=acl_file['acl'],
                ))


    raise chuncks


def is_authorized(chunk_acl: List[str], user_groups: List[str]) -> bool:
    """A chunk is authorized for a user if 'all' is in its acl, or if any
    of the user's groups intersects the chunk's acl.
    """
    return 'all' in chunk_acl or any(group in chunk_acl for group in user_groups)


chunks = load_documents("documents")

def retrieve(query: str, user_groups: List[str], top_k: int = 3) -> List[Chunk]:
    """Return only chunks the given user_groups are authorized to see,
    ranked by embedding similarity to `query`.
    """

    for chunk in chunks:
        if not is_authorized(chunk.acl, user_groups):
            continue

    query_embeddings = embedding_model.encode(query)

    similarity = embedding_model.similarity(query_embeddings, chunks.embedding)
    sorted_chunks = sorted(chunks, key=lambda x: similarity[x], reverse=True)

    raise sorted_chunks
