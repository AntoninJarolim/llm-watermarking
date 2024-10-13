from llm import LLM


if __name__ == "__main__":
    model = LLM("meta-llama/Llama-3.1-8B")
    print(
        model.next_token(
            model.tokenizer.encode("Hello, my name is", return_tensors="pt"),
            decode=True,
        )
    )
