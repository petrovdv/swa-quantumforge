import requests
import os
import mwparserfromhell

WIKI_API = "https://backtothefuture.fandom.com/api.php"

PAGES = [
    # герои
    "Marty_McFly",
    "Lorraine_Baines_McFly",
    "Jennifer_Parker",
    "George_McFly",
    "Emmett_Brown",
    "Clara_Clayton-Brown",
    "Jules_Brown",
    "Verne_Brown",
    "Biff_Tannen",
    # фильмы
    "Back_to_the_Future",
    "Back_to_the_Future_Part_II",
    "Back_to_the_Future_Part_III",
    "Back_to_the_Future:_The_Ride",
    # транспорт
    "DeLorean_Time_Machine",
    "Locomotive_131",
    "Toyota_Hilux",
    "Hoverboard",
    "Pulse",
    "Ford_Super_De_Luxe_Convertible",
    "Hover_conversion",
    # локации
    "Hill_Valley",
    "Hill_Valley_High_School",
    "Doc_Brown's_mansion",
    # "Lyon_Estates",
    "Doc's_garage",
    "Shonash_Ravine",
    "Hilldale",
    "Parker_Ranch",
    # события по годам
    "1885",
    "1955",
    "1985",
    "1985A",
    "1985B",
    "1985A-I",
    "2015",
]

OUTPUT_DIR = "../wiki_texts"

def fetch_page(title):
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "titles": title,
        "formatversion": 2,
    }

    response = requests.get(WIKI_API, params=params)
    data = response.json()

    try:
        return data["query"]["pages"][0]["revisions"][0]["content"]
    except Exception:
        print(f"[!] Ошибка загрузки: {title}")
        return None


def clean_wiki_text(wikitext):
    wikicode = mwparserfromhell.parse(wikitext)

    # Удаляем шаблоны (infobox, navbox и т.д.)
    for template in list(wikicode.filter_templates()):
        try:
            wikicode.remove(template)
        except ValueError:
            pass

    # Удаляем файлы/изображения
    for link in list(wikicode.filter_wikilinks()):
        if str(link.title).lower().startswith(("file:", "image:")):
            try:
                wikicode.remove(link)
            except ValueError:
                pass

    # Получаем чистый текст
    text = wikicode.strip_code()

    # Дополнительная чистка
    lines = []
    for line in text.split("\n"):
        line = line.strip()

        # пропускаем пустые и мусорные строки
        if not line:
            continue
        if line.startswith("|"):
            continue

        lines.append(line)

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for page in PAGES:
        print(f"Скачиваем: {page}")

        raw = fetch_page(page)
        if not raw:
            continue

        clean_text = clean_wiki_text(raw)

        filepath = os.path.join(OUTPUT_DIR, f"{page}.txt")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(clean_text)

    print("Готово!")


if __name__ == "__main__":
    main()