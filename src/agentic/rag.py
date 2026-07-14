from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from .config import settings


class SteelPilotRAG:

    def __init__(self, docs_dir: Path | None = None, vector_dir: Path | None = None):
        self.docs_dir = docs_dir or settings.docs_dir
        self.vector_dir = vector_dir or settings.vector_dir
        self.embeddings = None
        self.vectorstore: FAISS | None = None

    def _embeddings(self):
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        return self.embeddings

    def build(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        docs = loader.load()
        if not docs:
            raise RuntimeError(f"No markdown docs found in {self.docs_dir}. Run scripts/create_demo_docs.py first.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        self.vectorstore = FAISS.from_documents(chunks, self._embeddings())
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.vector_dir))

    def load(self) -> "SteelPilotRAG":
        if not self.vector_dir.exists():
            raise RuntimeError(f"FAISS index not found at {self.vector_dir}. Run python scripts/ingest_faiss.py first.")
        self.vectorstore = FAISS.load_local(
            str(self.vector_dir),
            self._embeddings(),
            allow_dangerous_deserialization=True,
        )
        return self

    def ensure_loaded(self) -> None:
        if self.vectorstore is None:
            self.load()

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        try:
            self.ensure_loaded()
            assert self.vectorstore is not None
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            out = []
            seen: set[tuple[str, str]] = set()
            for doc, score in results:
                src = doc.metadata.get("source", "unknown")
                item = {
                    "source": Path(src).name,
                    "score": float(score),
                    "content": doc.page_content.strip(),
                    "retrieval_mode": "faiss",
                }
                key = (item["source"], item["content"][:240])
                if key not in seen:
                    seen.add(key)
                    out.append(item)
            for item in self.keyword_retrieve(query, k=k):
                key = (item["source"], item["content"][:240])
                if key in seen:
                    continue
                out.append(item)
                seen.add(key)
                if len(out) >= k:
                    break
            return out[:k]
        except Exception:
            return self.keyword_retrieve(query, k=k)

    def keyword_retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        files = list(self.docs_dir.glob("**/*.md"))
        if not files:
            return []
        q_terms = [t for t in query.lower().replace("_", " ").split() if len(t) >= 4]
        scored = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            lower = text.lower()
            score = sum(lower.count(t) for t in q_terms)
            # small filename boost
            score += sum(str(f.name).lower().count(t) * 3 for t in q_terms)
            if score > 0:
                snippet = text.strip()[:1200]
                scored.append({"source": f.name, "score": float(score), "content": snippet, "retrieval_mode": "keyword_fallback"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]


SteelCareRAG = SteelPilotRAG
rag = SteelPilotRAG()
