from abc import ABC
from typing import Any, List, Optional

from langchain.embeddings import OpenAIEmbeddings

from langchaindemo.MagicAI.base import BaseEmbedding, num_tokens_from_string
from langchaindemo.MagicAI.open_ai.config import llmConfig


class MagicEmbed(BaseEmbedding):
    model: str = "text-embedding-ada-002"
    """specify the chatgpt model"""
    max_token: int = 4096
    """the max support token"""
    text: str = None
    """the embedding text"""
    texts: list = None
    llm: Any
    """Language model to call."""

    def _validate_embed(self):
        if not isinstance(self.text, str):
            raise TypeError("Invalid text type. Expected a non-string value.")
        # if check_max_token(self.text, self.max_token):
        #     raise ValueError("Tokens exceed the maximum limit")

    def _validate_embed_list(self):
        if not isinstance(self.texts, list):
            raise TypeError("Invalid text type. Expected a non-string value.")
        self.text = ''.join(text for text in self.texts)
        # if check_max_token(self.text, self.max_token):
        #     raise ValueError("Tokens exceed the maximum limit")

    def _validate_inputs(self, text: str = None, texts: list = None, **kwargs):
        self.model = kwargs.get('model', self.model)
        self.text = text
        self.texts = texts
        if self.text:
            self._validate_embed()
        elif self.texts:
            self._validate_embed_list()
        else:
            raise ValueError("Missing required keyword argument 'text' and 'texts'")
        chunk_size = kwargs.get("chunk_size",1000)
        if not isinstance(chunk_size,int):
            raise TypeError("The chunk_size type must be int,but you give the {}".format(type(chunk_size)))
        self.llm = OpenAIEmbeddings(openai_api_key=llmConfig.api_key, model=self.model,
                                    openai_proxy=llmConfig.openai_proxy,chunk_size=chunk_size)

    def _build_response_dict(self, embeddings) -> dict:
        prompt_tokens = num_tokens_from_string(self.text)
        completion_tokens = 0
        total_tokens = prompt_tokens + completion_tokens + 7
        response = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model": self.model, "embeddings": embeddings
        }
        return response

    def embeddings(self, text: str, **kwargs) -> dict:
        self._validate_inputs(text=text, **kwargs)
        embeddings = self.llm.embed_query(self.text)
        return self._build_response_dict(embeddings)

    def embed_documents(self, texts: List[str], **kwargs) -> dict:
        self._validate_inputs(texts=texts, **kwargs)
        embeddings = self.llm.embed_documents(self.texts)
        return self._build_response_dict(embeddings)


magicEmbed = MagicEmbed()
