from abc import ABC
from typing import List, Optional

from langchaindemo.MagicAI.base import AIBase
from langchaindemo.MagicAI.open_ai.chain.chain import magicChain
from langchaindemo.MagicAI.open_ai.embeddings.embed import magicEmbed


class OpenAI(AIBase, ABC):
    def _build_response_dict(self, **kwargs):
        pass

    def _validate_inputs(self, **kwargs):
        pass

    def run(self,
            template: str,
            input_variables: List = None,
            inputs: dict = None,
            **kwargs):
        return magicChain.run(template, input_variables, inputs, **kwargs)

    def json_schema(self,
                    json_schema: dict,
                    template: str,
                    input_variables: List[str] = None,
                    inputs: dict[str:str] = None,
                    **kwargs):
        return magicChain.json_schema(json_schema, template, input_variables, inputs, **kwargs)

    def chat(self, content, streaming, **kwargs):
        return magicChain.chat(content, streaming, **kwargs)

    def embeddings(self, text: str, **kwargs):
        return magicEmbed.embeddings(text, **kwargs)

    def embed_documents(self, texts: List[str], **kwargs):
        return magicEmbed.embed_documents(texts, **kwargs)
