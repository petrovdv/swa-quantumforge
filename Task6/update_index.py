"""
Скрипт обновления индекса для RAG-пайплайна.
Читает новые / изменённые TXT-файлы из NEW_DOCS_FOLDER,
обновляет эмбеддинги в ChromaDB и сохраняет манифест обработанных файлов.
Сейчас в конфигурации указаны относительные пути для тестового запуска.
При настройке запуска по крону пути нужно прописать полные.

Запуск по крону каждые 12 часов:
  0 */12 * * * /usr/bin/python3 update_index.py >> rag_index.log 2>&1
"""

import sys
import json
import hashlib
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- КОНФИГУРАЦИЯ ---
NEW_DOCS_FOLDER  = "./new_base"
CHROMA_DB_PATH   = "../Task3/chroma_db"
COLLECTION_NAME  = "text_files_collection"
MANIFEST_PATH    = "./indexer_manifest.json"  # state-файл обработанных документов
MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
BATCH_SIZE       = 100

def setup_logger() -> logging.Logger:
    """
    Логгер пишет в stdout с миллисекундами и именем модуля.
    При запуске через крон stdout перенаправляется в файл.
    """
    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("rag_indexer")


log = setup_logger()


def load_manifest(path: str) -> dict:
    """
    Возвращает словарь {filepath: {"hash": str, "chunk_ids": [str]}}
    """
    if not Path(path).exists():
        log.info("Манифест не найден, будет создан новый: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log.error("Повреждён манифест %s: %s. Создаём резервную копию.", path, e)
        Path(path).rename(f"{path}.bak.{int(datetime.now().timestamp())}")
        return {}


def save_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("Манифест сохранён: %s", path)


def file_md5(filepath: str) -> str:
    """MD5 содержимого файла — быстрый способ детектировать изменения."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_new_docs(folder: str, manifest: dict) -> tuple[list[dict], list[dict]]:
    """
    Сканирует папку и возвращает:
      new_files — файлы, которых нет в манифесте
      changed_files — файлы, чей хэш изменился
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        log.warning("Папка new_docs не существует: %s", folder)
        return [], []

    new_files, changed_files = [], []

    for filepath in sorted(folder_path.rglob("*.txt")):
        key = str(filepath)
        current_hash = file_md5(key)

        if key not in manifest:
            new_files.append({"path": key, "hash": current_hash})
        elif manifest[key]["hash"] != current_hash:
            changed_files.append({"path": key, "hash": current_hash})

    return new_files, changed_files


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def make_chunks(filepath: str, content: str, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    doc = Document(
        page_content=content,
        metadata={
            "source": filepath,
            "filename": Path(filepath).name,
            "original_length": len(content),
        },
    )
    return splitter.split_documents([doc])


def generate_chunk_ids(filepath: str, chunks: list[Document]) -> list[str]:
    """
    Детерминированные ID вида <md5_файла>_chunk_<N>
    Позволяют безопасно перезаписывать чанки при изменении файла.
    """
    file_hash = file_md5(filepath)
    return [f"{file_hash}_chunk_{i}" for i in range(len(chunks))]


def upsert_chunks(
    collection: chromadb.Collection,
    chunk_ids: list[str],
    chunks: list[Document],
    embeddings_list: list[list[float]],
) -> None:
    """
    Upsert (insert or update) — идемпотентная операция.
    ChromaDB перезапишет документы с совпадающими ID.
    """
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_ids   = chunk_ids[i : i + BATCH_SIZE]
        batch_docs  = chunks[i : i + BATCH_SIZE]
        batch_embs  = embeddings_list[i : i + BATCH_SIZE]

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=[c.page_content for c in batch_docs],
            metadatas=[c.metadata for c in batch_docs],
        )


def delete_old_chunks(collection: chromadb.Collection, old_chunk_ids: list[str]) -> None:
    """Удаляем устаревшие чанки изменённого файла перед переиндексацией."""
    if old_chunk_ids:
        collection.delete(ids=old_chunk_ids)
        log.info("  Удалено устаревших чанков: %d", len(old_chunk_ids))


def main() -> None:
    start_time = datetime.now(timezone.utc)
    log.info("=" * 60)
    log.info("СТАРТ индексации | %s", start_time.isoformat())
    log.info("=" * 60)

    errors: list[str] = []
    total_new_chunks = 0

    # 1. Загрузка манифеста и сканирование папки
    manifest = load_manifest(MANIFEST_PATH)
    new_files, changed_files = scan_new_docs(NEW_DOCS_FOLDER, manifest)

    log.info(
        "Обнаружено: новых файлов=%d, изменённых файлов=%d",
        len(new_files), len(changed_files),
    )

    if not new_files and not changed_files:
        log.info("Нет изменений — индексация не требуется.")
        _log_summary(start_time, 0, errors)
        return

    # 2. Инициализация инфраструктуры
    log.info("Загрузка модели эмбеддингов: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    log.info("Подключение к ChromaDB: %s", CHROMA_DB_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 3. Обработка изменённых файлов (сначала удаляем старые чанки)
    for file_info in changed_files:
        filepath = file_info["path"]
        filename = Path(filepath).name
        log.info("Переиндексация изменённого файла: %s", filename)
        try:
            old_chunk_ids = manifest.get(filepath, {}).get("chunk_ids", [])
            delete_old_chunks(collection, old_chunk_ids)

            content = read_file(filepath)
            chunks = make_chunks(filepath, content, splitter)
            chunk_ids = generate_chunk_ids(filepath, chunks)

            texts = [c.page_content for c in chunks]
            embeddings = model.encode(texts, show_progress_bar=False)
            embeddings_list = [e.tolist() for e in embeddings]

            upsert_chunks(collection, chunk_ids, chunks, embeddings_list)

            manifest[filepath] = {"hash": file_info["hash"], "chunk_ids": chunk_ids}
            total_new_chunks += len(chunks)
            log.info("  %s → %d чанков", filename, len(chunks))

        except Exception as exc:
            msg = f"Ошибка при обработке '{filename}': {exc}"
            log.error(msg)
            log.debug(traceback.format_exc())
            errors.append(msg)

    # 4. Обработка новых файлов
    for file_info in new_files:
        filepath = file_info["path"]
        filename = Path(filepath).name
        log.info("Индексация нового файла: %s", filename)
        try:
            content = read_file(filepath)
            chunks = make_chunks(filepath, content, splitter)
            chunk_ids = generate_chunk_ids(filepath, chunks)

            texts = [c.page_content for c in chunks]
            embeddings = model.encode(texts, show_progress_bar=False)
            embeddings_list = [e.tolist() for e in embeddings]

            upsert_chunks(collection, chunk_ids, chunks, embeddings_list)

            manifest[filepath] = {"hash": file_info["hash"], "chunk_ids": chunk_ids}
            total_new_chunks += len(chunks)
            log.info("  %s → %d чанков", filename, len(chunks))

        except Exception as exc:
            msg = f"Ошибка при обработке '{filename}': {exc}"
            log.error(msg)
            log.debug(traceback.format_exc())
            errors.append(msg)

    # 5. Сохранение манифеста
    save_manifest(MANIFEST_PATH, manifest)

    # 6. Итог
    index_size = collection.count()
    _log_summary(start_time, total_new_chunks, errors, index_size)


def _log_summary(
    start_time: datetime,
    new_chunks: int,
    errors: list[str],
    index_size: int | None = None,
) -> None:
    end_time = datetime.now(timezone.utc)
    elapsed  = (end_time - start_time).total_seconds()

    log.info("-" * 60)
    log.info("ЗАВЕРШЕНИЕ | %s", end_time.isoformat())
    log.info("Время выполнения      : %.2f сек", elapsed)
    log.info("Добавлено новых чанков: %d", new_chunks)
    if index_size is not None:
        log.info("Размер итогового индекса: %d чанков", index_size)
    if errors:
        log.error("Ошибок: %d", len(errors))
        for i, err in enumerate(errors, 1):
            log.error("  [%d] %s", i, err)
    else:
        log.info("Ошибок: 0")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical("Критическая ошибка: %s", exc)
        log.critical(traceback.format_exc())
        sys.exit(1)