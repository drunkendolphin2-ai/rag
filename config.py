import os

from dotenv import load_dotenv

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION = "warden_lore"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")

TOP_K = 4
# косинусное расстояние в Chroma: 0 - совпадение, 2 - противоположность.
# выше порога считаем, что в базе ничего релевантного нет
DISTANCE_THRESHOLD = 0.75

# слои защиты от промпт-инъекций (задание 5)
DEFENSE_PREPROMPT = True    # правило в системном промпте
DEFENSE_SANITIZE = True     # вырезание управляющих конструкций из чанков
DEFENSE_POSTCHECK = True    # отбрасывание подозрительных чанков целиком