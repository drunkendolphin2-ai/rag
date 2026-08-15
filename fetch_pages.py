import re
import time
from pathlib import Path

import requests

API = "https://naruto.fandom.com/api.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; rag-course-project/1.0)"}
OUT = Path("raw")
DELAY = 2.0

DROP_SECTIONS = {
    "trivia", "references", "external links", "see also", "gallery",
    "in other media", "quotes", "navigation", "notes", "video games",
    "creation and conception", "influence",
}

session = requests.Session()
session.headers.update(HEADERS)


# ---------- API ----------

def api_get(params: dict, attempts: int = 4) -> dict | None:
    for i in range(attempts):
        try:
            r = session.get(API, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"    сеть: {e}")
            time.sleep(5 * (i + 1))
            continue
        if r.status_code in (429, 503):
            wait = 10 * (i + 1)
            print(f"    {r.status_code}, пауза {wait}s")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}")
            return None
        try:
            return r.json()
        except ValueError:
            print("    ответ не JSON")
            time.sleep(5 * (i + 1))
    return None


def fetch_wikitext(title: str) -> str | None:
    data = api_get({
        "action": "parse", "page": title, "prop": "wikitext",
        "redirects": 1, "format": "json", "formatversion": 2,
    })
    if not data or "parse" not in data:
        return None
    return data["parse"].get("wikitext")


def resolve_title(title: str) -> str | None:
    data = api_get({
        "action": "query", "list": "search",
        "srsearch": title.replace("_", " "), "srlimit": 1,
        "format": "json", "formatversion": 2,
    })
    if not data:
        return None
    hits = data.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


# ---------- очистка вики-разметки ----------

LINK = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]+))?\]\]")
EXT_LINK = re.compile(r"\[(?:https?|//)\S+?(?:\s+([^\]]+))?\]")
FILE_LINK = re.compile(
    r"\[\[(?:File|Image|Media):[^\[\]]*(\[\[[^\]]*\]\])?[^\]]*\]\]", re.IGNORECASE
)
REF_TAG = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
HTML_TAG = re.compile(r"<[^>]+>")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
URL = re.compile(r"https?://\S+")
QUOTES = re.compile(r"'{2,5}")

# строки с именами файлов картинок - подписи к галереям, в базе знаний не нужны
IMG_LINE = re.compile(r"\S+\.(?:png|jpg|jpeg|gif|webp|svg)", re.IGNORECASE)


def strip_templates(text: str) -> str:
    """Убрать {{...}} с учётом вложенности."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i):
            depth = max(0, depth - 1)
            i += 2
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def strip_tables(text: str) -> str:
    out, depth = [], 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{|"):
            depth += 1
            continue
        if s.startswith("|}"):
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(line)
    return "\n".join(out)


def drop_sections(text: str) -> str:
    out, skip = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("==") and s.endswith("=="):
            name = s.strip("= ").strip().lower()
            skip = name in DROP_SECTIONS
            if skip:
                continue
            out.append(name.title())
            continue
        if not skip:
            out.append(line)
    return "\n".join(out)


def clean(text: str) -> str:
    text = COMMENT.sub("", text)
    text = REF_TAG.sub("", text)
    text = FILE_LINK.sub("", text)
    text = strip_templates(text)
    text = strip_tables(text)
    text = drop_sections(text)

    text = LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = EXT_LINK.sub(lambda m: m.group(1) or "", text)
    text = HTML_TAG.sub("", text)
    text = URL.sub("", text)
    text = QUOTES.sub("", text)

    lines = []
    for line in text.splitlines():
        s = line.strip()
        if IMG_LINE.search(s):
            continue
        if s.startswith(("*", "#", ":", ";")):
            s = s.lstrip("*#:; ").strip()
        if s in {"|", "}", "{"} or s.startswith("|"):
            continue
        lines.append(s)

    text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:])", r"\1", text)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# ---------- основной цикл ----------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    titles = [
        t.strip()
        for t in Path("pages.txt").read_text(encoding="utf-8").splitlines()
        if t.strip() and not t.startswith("#")
    ]

    ok, short, missing = 0, [], []
    for title in titles:
        target = OUT / f"{title}.txt"
        if target.exists():
            print(f"{title}: уже есть")
            ok += 1
            continue

        print(title)
        raw = fetch_wikitext(title)
        if not raw:
            found = resolve_title(title)
            if found and found != title:
                print(f"    -> найдено как «{found}»")
                raw = fetch_wikitext(found)

        if not raw:
            missing.append(title)
            print("    НЕ НАЙДЕНО")
            time.sleep(DELAY)
            continue

        text = clean(raw)
        if len(text) < 500:
            short.append(title)
            print(f"    КОРОТКО: {len(text)} символов")
        else:
            print(f"    {len(text)} символов")

        target.write_text(text, encoding="utf-8")
        ok += 1
        time.sleep(DELAY)

    print(f"\nСохранено: {ok} из {len(titles)}")
    if short:
        print("Коротких:", ", ".join(short))
    if missing:
        print("Не найдены:", ", ".join(missing))


if __name__ == "__main__":
    main()