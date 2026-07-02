import os

import litellm


def main():
    chat_completion = litellm.completion(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": "Write a short poem on OpenTelemetry.",
            },
        ],
    )
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()
