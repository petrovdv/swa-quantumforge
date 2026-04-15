import os

from fastapi import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import logging

# --- КОНФИГУРАЦИЯ ---
INPUT_FOLDER = "../Task2/knowledge_base/"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "text_files_collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


# --- ШАГ 1: Чтение всех TXT-файлов из папки ---
def load_text_files(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append({
                    'content': content,
                    'source': file_path,
                    'filename': filename
                })
    return documents


print("Чтение файлов из папки...")
raw_documents = load_text_files(INPUT_FOLDER)
print(f"Прочитано {len(raw_documents)} файлов")

# --- ШАГ 2: Разбиение на чанки с сохранением метаданных ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = []
chunk_id = 0

for doc in raw_documents:
    # Создаём документ LangChain с метаданными
    langchain_doc = Document(
        page_content=doc['content'],
        metadata={
            'source': doc['source'],
            'filename': doc['filename'],
            'original_length': len(doc['content'])
        }
    )

    # Разбиваем на чанки
    split_chunks = text_splitter.split_documents([langchain_doc])

    # Добавляем уникальный ID и обновляем метаданные для каждого чанка
    for chunk in split_chunks:
        chunk.metadata['chunk_id'] = chunk_id
        chunks.append(chunk)
        chunk_id += 1

print(f"Создано {len(chunks)} чанков")

# --- ШАГ 3: Генерация эмбеддингов ---
print("Загрузка модели эмбеддингов...")
model = SentenceTransformer(MODEL_NAME)

print("Генерация эмбеддингов...")
chunk_texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(chunk_texts, show_progress_bar=True)

print(f"Размер эмбеддинга: {embeddings[0].shape}")

# --- ШАГ 4: Создание и заполнение коллекции Chroma ---
print("Инициализация Chroma DB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Удаляем коллекцию, если она уже существует (для перезапуска)
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(name=COLLECTION_NAME)

print("Загрузка эмбеддингов в Chroma...")
# Пакетная загрузка для эффективности
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i + batch_size]
    batch_embeddings = embeddings[i:i + batch_size]

    ids = [f"chunk_{chunk.metadata['chunk_id']}" for chunk in batch_chunks]
    documents = [chunk.page_content for chunk in batch_chunks]
    metadatas = [chunk.metadata for chunk in batch_chunks]
    embeddings_list = [emb.tolist() for emb in batch_embeddings]

    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )

print(f"Успешно загружено {len(chunks)} чанков в коллекцию '{COLLECTION_NAME}'")

# --- ШАГ 5: Тестовый поиск ---
print("\n--- ТЕСТОВЫЙ ПОИСК ---")
test_query = "What do you know about Back to the Culture?"

# Генерируем эмбеддинг запроса
query_embedding = model.encode([test_query])

# Выполняем поиск
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5
)

# Выводим результаты
print(f"Результаты поиска по запросу: '{test_query}'")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
    print(f"\n--- Результат {i} ---")
    print(f"Файл: {meta['filename']}")
    print(f"Чанк ID: {meta['chunk_id']}")
    print(f"Текст: {doc[:300]}...")  # Первые 300 символов текста
