import re

# конструкции, которыми документ пытается управлять моделью
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous\s+|prior\s+|above\s+)?instructions?",
    r"disregard\s+(all\s+|the\s+)?(previous\s+|prior\s+|above\s+)?(instructions?|rules?)",
    r"forget\s+(everything|all|your\s+instructions?)",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?\s*:",
    r"system\s*:",
    r"</?(system|instruction|prompt)>",
    r"output\s*:\s*[\"«]",
    r"print\s+(the\s+)?(contents?|all)\s+of",
    r"reveal\s+(the\s+|your\s+)?(password|secret|system\s+prompt)",
    r"repeat\s+(the\s+|your\s+)?(system\s+prompt|instructions?)",
    r"act\s+as\s+(if|a|an)\s+",
    r"игнорируй\s+(все\s+)?(предыдущие\s+)?инструкции",
    r"забудь\s+(все|инструкции)",
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect(text: str) -> list[str]:
    """Вернуть список сработавших шаблонов."""
    return [m.group(0) for p in COMPILED for m in [p.search(text)] if m]


def sanitize(text: str) -> str:
    """Слой 2: вырезать управляющие конструкции, оставив остальной текст."""
    for pattern in COMPILED:
        text = pattern.sub("[REMOVED]", text)
    return text


def is_suspicious(text: str) -> bool:
    """Слой 3: чанк отбрасывается целиком."""
    return bool(detect(text))