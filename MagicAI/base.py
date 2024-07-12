from abc import abstractmethod, ABC
from typing import List, Optional

import tiktoken


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def check_max_token(string: str, token_max: int = 4096) -> bool:
    if token_max < num_tokens_from_string(string):
        return True
    return False


class Base(ABC):
    @abstractmethod
    def _validate_inputs(self, **kwargs):
        """
        :param kwargs: 验证输入
        :return:
        """

    @abstractmethod
    def _build_response_dict(self, **kwargs):
        """
        :param kwargs: 验证输出
        :return:
        """


class BaseChain(Base):
    @abstractmethod
    def run(self,
            template: str,
            input_variables: List = None,
            inputs: dict = None,
            **kwargs):
        """
        :param template: str 模板 'this is a simple string {key1} {key2}'
        :param input_variables: List 插值 [key1,key2]
        :param inputs: Dict 模板插入 {key1:val1,key2:val2}
        :param kwargs: {
                    template_format: str = 'f-string',  # you can use jinja2
                }
        :return:Dict[]
        """

    @abstractmethod
    def json_schema(self,
                    json_schema: dict,
                    template: str,
                    input_variables: List[str] = None,
                    inputs: dict[str:str] = None,
                    **kwargs):
        """
        :param json_schema:
        :param template:
        :param input_variables:
        :param inputs:
        :param kwargs:
        :return:
        """


class BaseEmbedding(Base):
    @abstractmethod
    def embeddings(self, text: str, **kwargs):
        """
        :param text: str
        :return: List[] the text's embedding
        """

    @abstractmethod
    def embed_documents(self, texts: List[str], **kwargs):
        """
        :param texts: List[]
        :return: List[List] the texts' embeddings
        """


class BaseChat(Base):
    @abstractmethod
    def chat(self, content: str, streaming: bool, **kwargs):
        """
        :param streaming:
        :param content:
        :return:
        """


class AIBase(BaseChain, BaseEmbedding, BaseChat, ABC):
    pass
