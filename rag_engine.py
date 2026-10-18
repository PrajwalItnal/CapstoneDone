"""
RAG Engine: Embedding-based Question Retrieval using FAISS + ChromaDB.
Implements semantic skill matching, question tagging, and retrieval-augmented generation
as required by the Enterprise Capstone project specification.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ─── Graceful imports ──────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_store"
try:
    CHROMA_DIR.mkdir(exist_ok=True)
except Exception as e:
    logger.warning(f"[RAG] Failed to create Chroma directory {CHROMA_DIR}: {e}")

# ─── Embedding model ───────────────────────────────────────────────────────────
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        except Exception as e:
            logger.error(f"[RAG] Could not load embedding model '{_EMBED_MODEL_NAME}': {e}")
    return _embed_model


def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """Return normalized float32 numpy array of shape (N, D) or None if unavailable."""
    if not texts:
        return None
    valid_texts = [str(t) if t is not None else "" for t in texts]
    if not any(valid_texts):
        return None

    model = get_embed_model()
    if model is None:
        return None
    try:
        vecs = model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
        vecs = vecs.astype(np.float32)
        if len(vecs.shape) == 1:
            vecs = np.expand_dims(vecs, axis=0)
        return vecs
    except Exception as e:
        logger.error(f"[RAG] Embedding error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# FAISS Index
# ════════════════════════════════════════════════════════════════════════════════

class FAISSIndex:
    """Flat cosine-similarity FAISS index for nearest-neighbour retrieval."""

    def __init__(self):
        self.index = None
        self.documents: List[Dict] = []
        self.dimension: int = 384

    def build(self, documents: List[Dict], text_field: str = "text") -> bool:
        if not documents:
            self.index = None
            self.documents = []
            return False

        if not FAISS_AVAILABLE or faiss is None:
            return False

        texts = [d.get(text_field, "") or d.get("question", "") or "" for d in documents]
        vecs = embed_texts(texts)
        if vecs is None or len(vecs) == 0:
            self.index = None
            self.documents = []
            return False

        try:
            dim = int(vecs.shape[1])
            faiss.normalize_L2(vecs)
            idx = faiss.IndexFlatIP(dim)
            idx.add(vecs)
            self.dimension = dim
            self.index = idx
            self.documents = list(documents)
            return True
        except Exception as e:
            logger.error(f"[RAG] Failed to build FAISS index: {e}")
            self.index = None
            self.documents = []
            return False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if not query or not str(query).strip():
            return []
        if self.index is None or self.index.ntotal == 0 or not self.documents:
            return []
        if top_k <= 0:
            return []

        qvec = embed_texts([str(query).strip()])
        if qvec is None:
            return []

        try:
            if FAISS_AVAILABLE and faiss is not None:
                faiss.normalize_L2(qvec)
            else:
                norm = np.linalg.norm(qvec, axis=1, keepdims=True)
                norm[norm == 0] = 1.0
                qvec = qvec / norm

            k = min(int(top_k), int(self.index.ntotal))
            scores, idxs = self.index.search(qvec, k)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx >= 0 and idx < len(self.documents):
                    results.append((float(score), self.documents[idx]))
            return results
        except Exception as e:
            logger.error(f"[RAG] FAISS search error: {e}")
            return []


# ════════════════════════════════════════════════════════════════════════════════
# ChromaDB Persistent Store
# ════════════════════════════════════════════════════════════════════════════════

_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if not CHROMA_AVAILABLE or chromadb is None:
        return None
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        except Exception as e:
            logger.error(f"[RAG] ChromaDB initialization error: {e}")
            _chroma_client = None
    return _chroma_client


def get_chroma_collection(collection_name: str = "question_bank"):
    client = get_chroma_client()
    if client is None:
        return None
    try:
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"[RAG] Failed to access Chroma collection '{collection_name}': {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# RAG Engine
# ════════════════════════════════════════════════════════════════════════════════

class RAGEngine:
    """
    Embedding-based retrieval engine for competency taxonomy and question bank.
    Implements semantic search (FAISS), persistent storage (ChromaDB), and
    multi-label tag recommendations as required by the capstone specification.
    """

    def __init__(self):
        self.question_index = FAISSIndex()
        self.taxonomy_index = FAISSIndex()
        self.taxonomy_docs: List[Dict] = []
        self.question_bank: List[Dict] = []
        self._indexed = False

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_question_bank(self, questions: List[Dict]) -> bool:
        """Index question dicts {question/text, competency, difficulty, ...} into FAISS + ChromaDB."""
        if not questions:
            self.question_bank = []
            self.question_index.build([])
            self._indexed = False
            return False

        self.question_bank = list(questions)
        docs = []
        texts = []
        for q in questions:
            q_text = str(q.get("question") or q.get("text") or "").strip()
            comp = str(q.get("competency") or "").strip()
            sub = str(q.get("subskill") or "").strip()
            tags_val = q.get("tags") or ""
            tags_str = " ".join(tags_val) if isinstance(tags_val, (list, tuple, set)) else str(tags_val)
            combined = f"{q_text} {comp} {sub} {tags_str}".strip() or "general question"

            doc = {**q, "text": combined, "question": q_text or combined}
            docs.append(doc)
            texts.append(combined)

        faiss_ok = self.question_index.build(docs, text_field="text")

        chroma_ok = False
        client = get_chroma_client()
        if client is not None:
            try:
                try:
                    client.delete_collection("question_bank")
                except Exception:
                    pass

                col = client.get_or_create_collection(
                    name="question_bank",
                    metadata={"hnsw:space": "cosine"}
                )

                if col is not None and texts:
                    vecs = embed_texts(texts)
                    embeddings = vecs.tolist() if vecs is not None else None

                    ids = []
                    seen_ids = set()
                    for idx, q in enumerate(questions):
                        base_id = str(q.get("id") or f"q_{idx}").strip()
                        unique_id = base_id
                        counter = 1
                        while unique_id in seen_ids:
                            unique_id = f"{base_id}_{counter}"
                            counter += 1
                        seen_ids.add(unique_id)
                        ids.append(unique_id)

                    metadatas = []
                    for q in questions:
                        meta = {}
                        for k, v in q.items():
                            if k not in ["text"]:
                                if isinstance(v, (str, int, float, bool)):
                                    meta[k] = v
                                elif v is None:
                                    meta[k] = ""
                                else:
                                    meta[k] = str(v)
                        metadatas.append(meta)

                    upsert_kwargs: Dict[str, Any] = {
                        "ids": ids,
                        "documents": texts,
                        "metadatas": metadatas,
                    }
                    if embeddings is not None:
                        upsert_kwargs["embeddings"] = embeddings

                    col.upsert(**upsert_kwargs)
                    chroma_ok = True
            except Exception as e:
                logger.error(f"[RAG] ChromaDB indexing error: {e}")

        self._indexed = bool(faiss_ok or chroma_ok)
        return self._indexed

    def index_taxonomy(self, taxonomy: Dict) -> bool:
        """Index the competency taxonomy for semantic skill mapping."""
        if not taxonomy or not isinstance(taxonomy, dict):
            self.taxonomy_docs = []
            self.taxonomy_index.build([])
            return False

        docs = []
        for skill, info in taxonomy.items():
            if not isinstance(info, dict):
                info = {}
            aliases = info.get("aliases", [])
            aliases_str = " ".join(aliases) if isinstance(aliases, (list, tuple)) else str(aliases or "")
            canonical = info.get("canonical", skill)
            category = info.get("category", "")
            text = f"{skill} {canonical} {aliases_str}".strip()
            docs.append({
                "skill": skill,
                "canonical": canonical,
                "category": category,
                "text": text or skill,
            })
        self.taxonomy_docs = docs
        return self.taxonomy_index.build(docs, text_field="text")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_questions(
        self,
        query: str,
        competency: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict]:
        """Retrieve semantically similar questions, optionally filtered."""
        if top_k is None or top_k <= 0:
            top_k = 5
        clean_query = str(query or "").strip()
        comp_filter = str(competency).strip().lower() if competency else None
        diff_filter = str(difficulty).strip().lower() if difficulty else None

        if not clean_query:
            return self._browse_fallback(comp_filter, diff_filter, top_k)

        # 1. Try FAISS retrieval first
        if self.question_index.index is not None and self.question_index.index.ntotal > 0:
            candidate_k = max(int(top_k) * 5, int(self.question_index.index.ntotal))
            faiss_results = self.question_index.search(clean_query, top_k=candidate_k)
            filtered = []
            for score, doc in faiss_results:
                if score < min_score:
                    continue
                doc_comp = str(doc.get("competency", "")).strip().lower()
                if comp_filter and doc_comp != comp_filter:
                    continue
                doc_diff = str(doc.get("difficulty", "")).strip().lower()
                if diff_filter and doc_diff != diff_filter:
                    continue
                filtered.append({**doc, "_score": round(float(score), 4)})
                if len(filtered) >= top_k:
                    break
            if filtered:
                return filtered

        # 2. Try ChromaDB retrieval second
        col = get_chroma_collection("question_bank")
        if col is not None and col.count() > 0:
            try:
                where_clause = None
                where_conditions = []
                if comp_filter:
                    where_conditions.append({"competency": {"$eq": str(competency)}})
                if diff_filter:
                    where_conditions.append({"difficulty": {"$eq": str(difficulty)}})

                if len(where_conditions) == 1:
                    where_clause = where_conditions[0]
                elif len(where_conditions) > 1:
                    where_clause = {"$and": where_conditions}

                query_vec = embed_texts([clean_query])
                fetch_n = min(max(top_k * 4, 10), col.count())
                query_kwargs: Dict[str, Any] = {"n_results": fetch_n}
                if where_clause:
                    query_kwargs["where"] = where_clause

                if query_vec is not None:
                    query_kwargs["query_embeddings"] = query_vec.tolist()
                else:
                    query_kwargs["query_texts"] = [clean_query]

                res = col.query(**query_kwargs)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]

                chroma_filtered = []
                for d, m, dist in zip(docs, metas, dists):
                    score = max(0.0, min(1.0, 1.0 - float(dist)))
                    if score < min_score:
                        continue
                    m_comp = str(m.get("competency", "")).strip().lower()
                    if comp_filter and m_comp != comp_filter:
                        continue
                    m_diff = str(m.get("difficulty", "")).strip().lower()
                    if diff_filter and m_diff != diff_filter:
                        continue
                    chroma_filtered.append({
                        **m,
                        "text": d,
                        "question": m.get("question") or d,
                        "_score": round(score, 4)
                    })
                    if len(chroma_filtered) >= top_k:
                        break

                if chroma_filtered:
                    return chroma_filtered
            except Exception as e:
                logger.error(f"[RAG] ChromaDB query error: {e}")

        # 3. Fallback to keyword search
        return self._keyword_fallback(clean_query, comp_filter, diff_filter, top_k, min_score)

    def _browse_fallback(self, comp_filter: Optional[str], diff_filter: Optional[str], top_k: int) -> List[Dict]:
        """Return non-empty questions matching filters when query string is blank."""
        results = []
        for q in self.question_bank:
            if comp_filter and str(q.get("competency", "")).strip().lower() != comp_filter:
                continue
            if diff_filter and str(q.get("difficulty", "")).strip().lower() != diff_filter:
                continue
            results.append({**q, "_score": 1.0})
            if len(results) >= top_k:
                break
        return results

    def _keyword_fallback(
        self,
        query: str,
        comp_filter: Optional[str],
        diff_filter: Optional[str],
        top_k: int,
        min_score: float = 0.0
    ) -> List[Dict]:
        keywords = set(str(query or "").lower().split())
        results = []
        for q in self.question_bank:
            doc_comp = str(q.get("competency", "")).strip().lower()
            if comp_filter and doc_comp != comp_filter:
                continue
            doc_diff = str(q.get("difficulty", "")).strip().lower()
            if diff_filter and doc_diff != diff_filter:
                continue

            text = f"{q.get('question', '')} {q.get('text', '')} {q.get('subskill', '')} {q.get('competency', '')}".lower()
            if keywords:
                overlap = sum(1 for kw in keywords if kw in text)
                score = overlap / max(len(keywords), 1)
            else:
                score = 0.5

            if score < min_score:
                continue

            results.append({**q, "_score": round(score, 4)})

        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:top_k]

    # ── Semantic Tagging ──────────────────────────────────────────────────────

    def recommend_tags(self, question_text: str, top_k: int = 3) -> List[Dict]:
        """Recommend top_k competency tags for a question using cosine similarity."""
        if not question_text or not str(question_text).strip():
            return []
        if self.taxonomy_index.index is None or self.taxonomy_index.index.ntotal == 0:
            return []
        if top_k <= 0:
            return []

        results = self.taxonomy_index.search(str(question_text).strip(), top_k=top_k * 2)
        recommended = []
        seen = set()
        for score, doc in results:
            canon = doc.get("canonical") or doc.get("skill") or ""
            canon = canon.strip()
            if not canon or canon.lower() in seen:
                continue
            if score > 0.25:
                seen.add(canon.lower())
                recommended.append({
                    "competency": canon,
                    "category": doc.get("category", ""),
                    "confidence": round(float(score), 3),
                })
            if len(recommended) >= top_k:
                break
        return recommended

    def detect_wrong_tags(
        self, question_text: str, assigned_competency: str, threshold: float = 0.35
    ) -> Dict:
        """Detect if a question is assigned to the wrong competency."""
        clean_assigned = str(assigned_competency or "").strip()
        tags = self.recommend_tags(question_text, top_k=1)
        if not tags:
            return {
                "is_wrong": False,
                "suggested_competency": clean_assigned,
                "confidence": 0.0,
            }
        top_tag = tags[0]
        suggested = top_tag["competency"]
        confidence = float(top_tag["confidence"])

        is_wrong = (
            suggested.strip().lower() != clean_assigned.lower()
            and confidence >= threshold
        )
        return {
            "is_wrong": is_wrong,
            "suggested_competency": suggested,
            "confidence": confidence,
        }

    # ── Competency Mapping ────────────────────────────────────────────────────

    def map_skill_to_competency(self, skill_text: str) -> Optional[Dict]:
        """Map a raw extracted skill to canonical competency via semantic similarity."""
        if not skill_text or not str(skill_text).strip():
            return None
        if self.taxonomy_index.index is None or self.taxonomy_index.index.ntotal == 0:
            return None

        results = self.taxonomy_index.search(str(skill_text).strip(), top_k=1)
        if results:
            score, doc = results[0]
            if score > 0.25:
                canon = doc.get("canonical") or doc.get("skill") or ""
                return {
                    "canonical": canon.strip(),
                    "category": doc.get("category", ""),
                    "confidence": round(float(score), 3),
                }
        return None

    # ── Duplicate Detection ───────────────────────────────────────────────────

    def find_duplicates(
        self, questions: List[Dict], similarity_threshold: float = 0.85
    ) -> List[Tuple[int, int, float]]:
        """Find pairs of questions that are too similar (potential duplicates)."""
        if not questions or len(questions) < 2:
            return []

        texts = [str(q.get("question") or q.get("text") or "").strip() for q in questions]
        vecs = embed_texts(texts)
        if vecs is None or len(vecs) < 2:
            return []

        if FAISS_AVAILABLE and faiss is not None:
            faiss.normalize_L2(vecs)
        else:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms

        duplicates = []
        num_vecs = len(vecs)
        for i in range(num_vecs):
            for j in range(i + 1, num_vecs):
                sim = float(np.dot(vecs[i], vecs[j]))
                if sim >= similarity_threshold:
                    duplicates.append((i, j, round(sim, 3)))
        return duplicates

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        q_count = self.question_index.index.ntotal if self.question_index.index else 0
        t_count = self.taxonomy_index.index.ntotal if self.taxonomy_index.index else 0
        col = get_chroma_collection("question_bank")
        chroma_count = 0
        if col:
            try:
                chroma_count = col.count()
            except Exception:
                pass
        return {
            "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
            "faiss": FAISS_AVAILABLE,
            "chromadb": CHROMA_AVAILABLE,
            "indexed": self._indexed,
            "questions_indexed_faiss": q_count,
            "taxonomy_indexed_faiss": t_count,
            "questions_in_chromadb": chroma_count,
            "model": _EMBED_MODEL_NAME,
        }

    get_status = status


# ─── Singleton ─────────────────────────────────────────────────────────────────
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
