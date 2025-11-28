import re
from typing import List, Dict, Any
from docx import Document
from pathlib import Path
import json

def extract_articles_from_docx(file_path: str) -> List[Dict[str, Any]]:
    doc = Document(file_path)
    articles = []
    current_article = None
    current_section = ""
    current_chapter = ""

    paragraphs = doc.paragraphs
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        text = para.text.strip()
        if not text:
            i += 1
            continue

        # Обработка раздела
        if re.match(r"^РАЗДЕЛ\s+\S+", text.upper()):
            current_section = text
            i += 1
            continue

        # Обработка главы
        if re.match(r"^ГЛАВА\s+\d+", text.upper()):
            current_chapter = text
            # Проверяем следующую строку — возможно, это продолжение названия главы
            if i + 1 < len(paragraphs):
                next_text = paragraphs[i + 1].text.strip()
                if next_text and not re.match(r"^(Статья\s+\d+|РАЗДЕЛ|ГЛАВА)", next_text, re.IGNORECASE):
                    current_chapter += " " + next_text
                    i += 2  # пропускаем обе строки
                else:
                    i += 1
            else:
                i += 1
            continue

        # Новая статья
        article_match = re.match(r"^Статья\s+(\d+)", text, re.IGNORECASE)
        if article_match:
            if current_article is not None:
                articles.append(current_article)

            current_article = {
                "article_number": article_match.group(1),
                "section": current_section,
                "chapter": current_chapter,
                "content": "",
                "metadata": {
                    "source": Path(file_path).name,
                    "type": "constitution_article"
                }
            }
            i += 1
            continue

        # Добавление содержания статьи
        if current_article is not None:
            if current_article["content"]:
                current_article["content"] += "\n" + text
            else:
                current_article["content"] = text
        i += 1

    # Добавляем последнюю статью
    if current_article is not None:
        articles.append(current_article)

    return articles

def save_articles_to_json(articles: List[Dict[str, Any]], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"Статьи сохранены в {output_path}")

if __name__ == "__main__":
    docx_path = "constitutionrf.docx"
    output_json = "data/constitution_articles.json"

    print("Извлечение статей из .docx файла...")
    articles = extract_articles_from_docx(docx_path)
    print(f"Найдено статей: {len(articles)}")

    Path("data").mkdir(exist_ok=True)
    save_articles_to_json(articles, output_json)

    if articles:
        print("\nПример первой статьи:")
        print(f"Статья {articles[0]['article_number']}")
        print(f"Раздел: {articles[0]['section']}")
        print(f"Глава: {articles[0]['chapter']}")
        print(f"Содержание:\n{articles[0]['content'][:200]}...")
    else:
        print("❗ ОШИБКА: Не найдено ни одной статьи. Проверьте структуру .docx файла")