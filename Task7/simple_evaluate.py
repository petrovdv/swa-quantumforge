"""
Оценка RAG без LLM-метрик (только embedding-based / similarity подход).
"""

import json
import logging
import requests
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
import numpy as np


KEYS_FILE = "golden_set.json"
API_URL   = "http://localhost:8000/ask"
LOG_FILE  = "rag_evaluation.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def load_questions(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    log.info(f"Загружено {len(data)} вопросов")
    return data


def ask(query: str) -> dict | None:
    try:
        resp = requests.post(API_URL, json={"query": query}, timeout=360)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"[{query}] error: {e}")
        return None


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    qa_pairs = load_questions(KEYS_FILE)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    scores = []

    for question, ground_truth in qa_pairs.items():
        log.info(f"→ {question}")

        response = ask(question)
        if not response:
            continue

        answer = response.get("answer", "")

        # embeddings
        q_emb = embeddings.embed_query(question)
        a_emb = embeddings.embed_query(answer)
        gt_emb = embeddings.embed_query(ground_truth)

        # similarity metrics
        answer_relevance = cosine(q_emb, a_emb)
        correctness      = cosine(a_emb, gt_emb)

        scores.append({
            "question": question,
            "answer_relevancy": answer_relevance,
            "answer_correctness": correctness,
        })

    # summary
    avg_relevancy = np.mean([s["answer_relevancy"] for s in scores])
    avg_correct   = np.mean([s["answer_correctness"] for s in scores])

    log.info("=" * 60)
    log.info(f"AVG relevancy   : {avg_relevancy:.4f}")
    log.info(f"AVG correctness : {avg_correct:.4f}")

    output_path = Path("ragas_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    log.info(f"Saved to {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()