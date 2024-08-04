import json
import math
import os
from typing import Dict, List

from openai import OpenAI

from utils.json_tokenizer import JsonTokenizer


def softmax(tokens: List[Dict[str, float]]) -> List[Dict[str, float]]:
    exp_probs = [{"token": t["token"], "prob": math.exp(t["logprob"])} for t in tokens]
    total = sum(t["prob"] for t in exp_probs)
    return [{"token": t["token"], "prob": t["prob"] / total} for t in exp_probs]


def preprocessor(
    tokens: List[Dict[str, float]], json_tokenizer: JsonTokenizer
) -> List[Dict[str, float]]:
    valid_tokens = [t for t in tokens if json_tokenizer.is_valid(t["token"])]

    if not valid_tokens:
        if json_tokenizer.stack:
            closing_token = json_tokenizer.stack[-1]
            if json_tokenizer.is_valid(closing_token):
                valid_tokens = [
                    {"token": closing_token, "logprob": -10.0}
                ]  # 给一个很小的概率
        elif not json_tokenizer.has_content:
            valid_tokens = [
                {"token": "{", "logprob": -1.0},
                {"token": "[", "logprob": -1.0},
            ]

    return softmax(valid_tokens)


def generate_json_with_llm(prompt: str, max_tokens: int = 100) -> str:
    json_tokenizer = JsonTokenizer()
    result = ""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    while len(result) < max_tokens and not json_tokenizer.is_complete():
        print(f"Prompt: {prompt + result}")  # 为了调试
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON generator. Generate valid JSON only.",
                },
                {"role": "user", "content": prompt + result},
            ],
            max_tokens=1,  # 每次只生成一个 token
            n=1,
            temperature=0.7,
            logprobs=True,
            top_logprobs=5,
        )

        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            token_info = response.choices[0].logprobs.content[0]
            raw_tokens = [
                {"token": logprob.token, "logprob": logprob.logprob}
                for logprob in token_info.top_logprobs
            ]

            processed_tokens = preprocessor(raw_tokens, json_tokenizer)

            if processed_tokens:
                next_token = max(processed_tokens, key=lambda x: x["prob"])
                result += next_token["token"]
                json_tokenizer.is_valid(next_token["token"])
                print(f"Generated token: {next_token['token']}")  # 为了调试
            else:
                print("No valid tokens available. Ending generation.")
                break

        prompt += f"\nPlease continue writing the answer in json format:\n{result}"

    return result


if __name__ == "__main__":
    prompt = "Task: Generate a JSON object describing a person with name and age. The answer schema is as follows:"
    schema = {
        "name": "string",
        "age": "number",
    }

    print("Generating JSON...")
    generated_json = generate_json_with_llm(
        prompt + "\n" + json.dumps(schema, indent=2)
    )
    print("\nGenerated JSON:")
    print(generated_json)

    try:
        parsed_json = json.loads(generated_json)
        print("\nSuccessfully parsed the generated JSON:")
        print(json.dumps(parsed_json, indent=2))
    except json.JSONDecodeError as e:
        print(f"\nError: Generated JSON is not valid: {e}")
