import json
import re
from pathlib import Path

RAW = Path("raw")
OUT = Path("knowledge_base")


def restore_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def main() -> None:
    OUT.mkdir(exist_ok=True)
    terms_map = json.loads(Path("terms_map.json").read_text(encoding="utf-8"))

    lookup = {k.lower(): v for k, v in terms_map.items()}
    ordered = sorted(lookup, key=len, reverse=True)

    # (s|es)? ловит множественное число: Hokages, Uchihas, chakras
    pattern = re.compile(
        r"(?<![\w'])(" + "|".join(re.escape(t) for t in ordered) + r")(s|es)?(?![\w])",
        re.IGNORECASE,
        )

    def sub(m: re.Match) -> str:
        found = m.group(1)
        suffix = m.group(2) or ""
        return restore_case(found, lookup[found.lower()]) + suffix

    total = 0
    for src in sorted(RAW.glob("*.txt")):
        text = pattern.sub(sub, src.read_text(encoding="utf-8"))

        # подчёркивания мешают границе слова, поэтому меняем их на пробелы
        stem = src.stem.replace("_", " ")
        name = pattern.sub(sub, stem).replace(" ", "_")

        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        total += 1
        print(f"{src.name} -> {name}.txt")

    print(f"\nОбработано: {total}")


if __name__ == "__main__":
    main()