import asyncio
from abc import ABC

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from langchaindemo.MagicAI.base import BaseChat, check_max_token, num_tokens_from_string
from langchaindemo.MagicAI.custom import StreamingWeb
from langchaindemo.MagicAI.open_ai.config import llmConfig

prompt_tokens: int = 0
completion_tokens: int = 0
total_tokens: int = 0


class ChatChain(BaseChat, ABC):
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    """the random value used to generate"""
    max_token: int = 4096
    streaming: bool = False
    content: str = ""

    def _validate_inputs(self, content: str, streaming: bool = False, **kwargs):
        if content is None:
            raise ValueError("Missing required keyword argument 'content'")

        if check_max_token(content):
            raise ValueError("Tokens exceed the maximum limit")

        self.content = content

        self.streaming = streaming

        self.model: str = kwargs.get('model', self.model)
        self.temperature: float = kwargs.get('temperature', self.temperature)

    def _build_response_dict(self, **kwargs):
        pass

    def _exec_steaming_chat(self):
        callback_handler = StreamingWeb()
        chatllm = ChatOpenAI(openai_api_key=llmConfig.api_key,
                             openai_proxy=llmConfig.openai_proxy,
                             model=self.model,
                             temperature=self.temperature,
                             streaming=True,
                             callbacks=[callback_handler])
        content_tokens = num_tokens_from_string(self.content)
        res = chatllm([HumanMessage(content=self.content)])
        for response in callback_handler.generate_tokens(content_tokens):
            yield response

    def _exec_chat(self):
        chat = ChatOpenAI(openai_api_key=llmConfig.api_key, openai_proxy=llmConfig.openai_proxy,
                          model=self.model,
                          temperature=self.temperature, streaming=False)
        global completion_tokens, prompt_tokens, total_tokens
        prompt_tokens = num_tokens_from_string(self.content)
        resp = chat([HumanMessage(content=self.content)])
        completion_tokens = num_tokens_from_string(resp.content)
        total_tokens = prompt_tokens + completion_tokens + 7

        response = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model": self.model, "answer": resp.content
        }
        return response

    async def _exec_achat(self) -> None:
        global prompt_tokens, completion_tokens, total_tokens
        prompt_tokens = num_tokens_from_string(self.content)
        total_tokens = prompt_tokens
        stream_handler = AsyncIteratorCallbackHandler()
        chat = ChatOpenAI(openai_api_key=llmConfig.api_key, streaming=True, callbacks=[stream_handler],
                          temperature=self.temperature, model=self.model,
                          openai_proxy=llmConfig.openai_proxy)
        asyncio.create_task(chat.agenerate([[HumanMessage(content=self.content)]]))

        # 使用异步迭代器遍历结果
        async for token in stream_handler.aiter():
            completion_tokens = num_tokens_from_string(token)
            total_tokens += completion_tokens

            response = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens + 7,
                "answer": token
            }
            yield response  # 使用yield语句返回每个response

    async def achat(self, content: str, streaming: bool = False, **kwargs):
        """
        可以使用这样字调用chat 进行流式输出
        async def receive_and_print():
        async for response in chatChain.chat(content="hello who are you",streaming=True):
            print(response)

        if __name__ == "__main__":
            loop = asyncio.get_event_loop()
            loop.run_until_complete(receive_and_print())

        :param content:
        :param streaming:
        :param kwargs:
        :return:
        """
        self._validate_inputs(content, streaming, **kwargs)
        if streaming:
            async for response in self._exec_achat():
                yield response
        else:
            yield self._exec_chat()

    def chat(self, content: str, streaming: bool = False, **kwargs) -> None:
        self._validate_inputs(content, streaming, **kwargs)
        if streaming:
            for response in self._exec_steaming_chat():
                yield response
        else:
            yield self._exec_chat()


chatChain = ChatChain()
