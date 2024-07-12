import time
from typing import Any, Generator

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult


class StreamingWeb(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.completion_tokens = 0
        self.total_tokens = 0
        self.tokens = []
        self.finish = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.finish = True

    def generate_tokens(self, prompt_tokens) -> Generator:
        while not self.finish or self.tokens:
            if self.tokens:
                data = self.tokens.pop(0)
                self.completion_tokens += 1
                self.total_tokens = self.completion_tokens + prompt_tokens

                response = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens + 7,
                    "answer": data
                }
                yield response
            else:
                time.sleep(1)
