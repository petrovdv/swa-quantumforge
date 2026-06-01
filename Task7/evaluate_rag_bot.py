"""
Извлекает вопросы и эталонные ответы из JSON-файла,
отправляет POST-запросы на /ask,
оценивает ответы с помощью RAGAS и пишет результаты в лог.
"""

import json
import logging
import requests
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas import evaluate
from datasets import Dataset



KEYS_FILE = "golden_set.json"          # файл с вопросами
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
    """Считывает вопросы и эталонные ответы из JSON (dict)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Ожидался dict, получен {type(data).__name__}")

    log.info(f"Загружено {len(data)} вопросов из «{path}»")
    return data


def ask(query: str) -> dict | None:
    """POST /ask → возвращает JSON-ответ или None при ошибке."""
    payload = {"query": query}
    try:
        # запрос может выполняться долго
        resp = requests.post(API_URL, json=payload, timeout=360)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        log.error(f"[{query!r}] Не удалось подключиться к {API_URL}")
    except requests.exceptions.Timeout:
        log.error(f"[{query!r}] Таймаут запроса")
    except requests.exceptions.HTTPError as e:
        log.error(f"[{query!r}] HTTP-ошибка: {e}")
    except Exception as e:
        log.error(f"[{query!r}] Неизвестная ошибка: {e}")
    return None


def build_ragas_dataset(records: list[dict]) -> Dataset:
    """
    Формирует датасет для RAGAS:
      { question, answer, contexts, ground_truth }
    """
    return Dataset.from_dict({
        "question":     [r["question"] for r in records],
        "answer":       [r["answer"] for r in records],
        "contexts":     [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })


def run_ragas(dataset: Dataset):
    llm = ChatOllama(
        model="phi3:mini",
        temperature=0,
        num_ctx=2048,
    )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    return result

def main():
    log.info("=" * 60)
    log.info(f"Старт оценки — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # 1. Загружаем вопросы и эталонные ответы
    qa_pairs = load_questions(KEYS_FILE)

    # 2. Собираем ответы с сервера
    records = []
    for question, ground_truth in qa_pairs.items():
        log.info(f"→ Запрос: {question!r}")
        response = ask(question)

        if response is None:
            log.warning(f"  Пропускаем вопрос (нет ответа от сервера)")
            continue

        answer   = response.get("answer", "")
        contexts = response.get("context", [])   # список строк

        log.info(f"  Ответ получен ({len(answer)} символов, {len(contexts)} контекстов)")

        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    if not records:
        log.error("Нет данных для оценки — завершение.")
        return

    # 3. RAGAS-оценка
    log.info("-" * 60)
    log.info(f"Запуск RAGAS для {len(records)} вопросов…")

    dataset = build_ragas_dataset(records)
    scores  = run_ragas(dataset)

    # 4. Логируем общие метрики
    log.info("=" * 60)
    log.info("RAGAS — сводные метрики:")
    for metric, value in scores.items():
        log.info(f"  {metric:<25} {value:.4f}")

    # 5. Логируем построчные результаты
    log.info("-" * 60)
    log.info("Детализация по вопросам:")

    scores_df = scores.to_pandas()
    for i, (_, row) in enumerate(scores_df.iterrows()):
        q = records[i]["question"]
        log.info(f"\n  [{i+1}] {q!r}")
        log.info(f"       faithfulness        = {row.get('faithfulness', float('nan')):.4f}")
        log.info(f"       answer_relevancy    = {row.get('answer_relevancy', float('nan')):.4f}")
        log.info(f"       answer_correctness  = {row.get('answer_correctness', float('nan')):.4f}")

    # 6. Сохраняем результаты в JSON
    output_path = Path("ragas_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    log.info(f"\nРезультаты сохранены в «{output_path}»")
    log.info("=" * 60)


if __name__ == "__main__":
    main()