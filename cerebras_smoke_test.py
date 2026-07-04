from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from cerebras.cloud.sdk import Cerebras
except ModuleNotFoundError:
    print("FAIL: cerebras-cloud-sdk is not installed in this environment.")
    sys.exit(1)


def main() -> int:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=True)

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("FAIL: CEREBRAS_API_KEY is not set.")
        return 1

    model = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")

    try:
        client = Cerebras(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Reply with exactly: Cerebras is working"}
            ],
            max_completion_tokens=128,
            temperature=0.0,
            top_p=1,
            stream=False,
            reasoning_effort="medium",
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            print("FAIL: Cerebras request succeeded but returned empty content.")
            return 1
        print("PASS: Cerebras request succeeded.")
        print(f"Model: {model}")
        print(f"Response: {content}")
        return 0
    except Exception as exc:
        print("FAIL: Cerebras request failed.")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
