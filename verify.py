import re
from collections import Counter
from pathlib import Path

OUT = Path("knowledge_base")

FORBIDDEN = [
    "Naruto", "Sasuke", "Sakura", "Kakashi", "Itachi", "Madara", "Obito",
    "Hinata", "Neji", "Shikamaru", "Minato", "Gaara", "Jiraiya", "Tsunade",
    "Orochimaru", "Kurama", "Uchiha", "Hyuga", "Hyūga", "Senju", "Uzumaki",
    "Konoha", "Suna", "Kiri", "Kumo", "Iwa", "chakra", "jutsu", "shinobi",
    "ninja", "Hokage", "Kage", "Sharingan", "Byakugan", "Rinnegan",
    "Rasengan", "Chidori", "Akatsuki", "Bijuu", "Jinchuriki", "Jinchūriki",
    "genin", "chunin", "jonin", "kunai", "shuriken", "Susanoo",
]

# короткие термины ищем только как отдельные слова -
# иначе ложные срабатывания внутри tsunami, package, Sakumo и т.п.
SHORT = {"iwa", "suna", "kumo", "kiri", "kage", "naruto", "neji"}


def build_pattern() -> re.Pattern:
    parts = []
    for t in FORBIDDEN:
        esc = re.escape(t)
        parts.append(rf"\b{esc}\b" if t.lower() in SHORT else esc)
    return re.compile("|".join(parts), re.IGNORECASE)


def main() -> None:
    pattern = build_pattern()
    found: Counter = Counter()
    samples: dict[str, list[str]] = {}

    docs = sorted(OUT.glob("*.txt"))
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            hit = m.group(0)
            found[hit] += 1
            ctx = text[max(0, m.start() - 45):m.end() + 45].replace("\n", " ")
            samples.setdefault(hit, [])
            if len(samples[hit]) < 3:
                samples[hit].append(ctx.strip())

    print(f"Документов: {len(docs)}\n")
    for term, n in found.most_common():
        print(f"{term} ({n}):")
        for ctx in samples[term]:
            print(f"    ...{ctx}...")
        print()

    if not found:
        print("Исходных терминов не найдено")
    else:
        print(f"Всего вхождений: {sum(found.values())}")


if __name__ == "__main__":
    main()