import requests

r = requests.get(
    "https://naruto.fandom.com/api.php",
    params={
        "action": "parse",
        "page": "Gaara",
        "prop": "wikitext",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    },
    headers={"User-Agent": "Mozilla/5.0 (rag-course-project)"},
    timeout=30,
)
print("HTTP:", r.status_code)
data = r.json()
print("Ключи:", list(data.keys()))
if "parse" in data:
    print(data["parse"]["wikitext"][:500])
else:
    print(data)