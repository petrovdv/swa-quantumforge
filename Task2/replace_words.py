import os
import json
import re

# Параметры
SOURCE_DIRECTORY = "wiki_texts"  # Исходная директория с текстовыми файлами
TARGET_DIRECTORY = "knowledge_base"  # Целевая директория для скопированных файлов
CONFIG_FILE = "terms_map.json"  # Файл конфигурации с заменами


def preserve_case_replace(text, old_word, new_word):
    """
    Заменяет все вхождения old_word на new_word без учёта регистра,
    сохраняя исходный регистр в тексте.
    """
    def replace_match(match):
        # Получаем совпавшее слово с оригинальным регистром
        matched_text = match.group(0)
        # Определяем регистр совпавшего слова
        if matched_text.isupper():
            return new_word.upper()
        elif matched_text.islower():
            return new_word.lower()
        elif matched_text.istitle():  # Первая буква заглавная
            return new_word.capitalize()
        else:
            # Для сложных случаев (camelCase и т. п.) — сохраняем первую букву как в оригинале
            if matched_text[0].isupper():
                return new_word.capitalize()
            else:
                return new_word.lower()

    # Создаём регулярное выражение с флагом IGNORECASE для поиска без учёта регистра
    pattern = re.compile(re.escape(old_word), re.IGNORECASE)
    return pattern.sub(replace_match, text)

def replace_words_in_file(input_file_path, output_file_path, replacements):
    """
    Читает входной файл, заменяет слова согласно словарю replacements
    и записывает результат в выходной файл.
    """
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Выполняем замены, сохраняя регистр
    for old_word, new_word in replacements.items():
        content = preserve_case_replace(content, old_word, new_word)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_directory(source_dir, target_dir, config_file):
    """
    Проходит по всем текстовым файлам в source_dir,
    копирует их в target_dir с заменой слов из config_file.
    """
    # Создаём целевую директорию, если её нет
    os.makedirs(target_dir, exist_ok=True)

    # Загружаем словарь замен из JSON‑файла
    with open(config_file, 'r', encoding='utf-8') as f:
        replacements = json.load(f)

    # Проходим по всем файлам в исходной директории
    for filename in os.listdir(source_dir):
        source_file_path = os.path.join(source_dir, filename)

        # Обрабатываем только файлы (не директории)
        if os.path.isfile(source_file_path):
            # Определяем путь к целевому файлу
            target_file_path = os.path.join(target_dir, filename)

            # Копируем и заменяем слова в текстовых файлах
            replace_words_in_file(source_file_path, target_file_path, replacements)
            print(f"Обработан файл: {filename}")


if __name__ == "__main__":
    process_directory(SOURCE_DIRECTORY, TARGET_DIRECTORY, CONFIG_FILE)
