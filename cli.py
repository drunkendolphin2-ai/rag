from rag import RagBot


def main() -> None:
    print("Загрузка...")
    bot = RagBot()
    print("Готово. Пустая строка или exit - выход.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break

        answer = bot.ask(question)

        if answer.reasoning:
            print(f"\n[рассуждение]\n{answer.reasoning}")
        print(f"\n{answer.text}")

        if answer.sources:
            print("\nИсточники:")
            for s in answer.sources:
                print(f"  {s.title} (чанк {s.chunk_index}, "
                      f"расстояние {s.distance})")
        else:
            print("\nРелевантных фрагментов в базе не найдено.")
        print()


if __name__ == "__main__":
    main()