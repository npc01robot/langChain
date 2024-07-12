from typing import List, Dict, Union, Any, Optional

from langchain import PromptTemplate, BasePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel

from langchaindemo.MagicAI.base import BaseChain, check_max_token
from langchaindemo.MagicAI.custom.magic_llm_chain import MagicLLMChain
from langchaindemo.MagicAI.open_ai.chain.base import jsonschema_output_parser
from langchaindemo.MagicAI.open_ai.config import llmConfig


class MagicChain(BaseChain):
    temperature: float = 0.7
    """the random value used to generate"""
    n: int = 1
    """used to generate the number of returns """
    model: str = "gpt-3.5-turbo"
    """specify the chatgpt model"""
    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    """Language model to call."""
    template: str = None
    """the input template"""
    input_variables: List = None
    """the match template list"""
    inputs: dict = None
    """the match inputs"""
    max_token: int = 4096
    """the max support token"""

    # 验证输入

    def _validate_inputs(self, template, input_variables, inputs, **kwargs):
        self.template = template
        if not self.template:
            raise ValueError("Missing required keyword argument 'template'")

        self.input_variables = input_variables
        if not self.input_variables:
            self.template += "{temp}"
            self.input_variables = ["temp"]

        self.inputs = inputs
        if not self.inputs:
            self.inputs = {"temp": ""}

        self.n = kwargs.get("n", self.n)
        if not self.n:
            raise ValueError("Missing required keyword argument 'n'")

        self.model: str = kwargs.get('model', self.model)

        self.temperature: float = kwargs.get('temperature', self.temperature)

        self.template_format: str = kwargs.get('template_format', "f-string")

        self.prompt = PromptTemplate(template=self.template, input_variables=self.input_variables,
                                     template_format=self.template_format)
        self.llm = ChatOpenAI(openai_api_key=llmConfig.api_key, openai_proxy=llmConfig.openai_proxy, model=self.model,
                              temperature=self.temperature)
        text = self.prompt.format_prompt(**self.inputs).text
        if check_max_token(text, self.max_token):
            raise ValueError("The Input tokens exceed the maximum limit")

    def _build_response_dict(self, chain_results: dict) -> dict[str, Union[Optional[str], Any]]:
        response = {
            "prompt_tokens": chain_results.get("prompt_tokens"),
            "completion_tokens": chain_results.get("completion_tokens"),
            "total_tokens": chain_results.get("total_tokens"),
            "model": self.model, "answer": chain_results.get("result")
        }
        return response

    # 基本输出格式
    def run(self, template: str,
            input_variables: List = None,
            inputs: dict = None,
            **kwargs) -> dict[str, Union[Optional[str], Any]]:
        self._validate_inputs(template, input_variables, inputs, **kwargs)

        chain = MagicLLMChain(llm=self.llm, prompt=self.prompt, n=self.n)

        chain_results = chain.run(self.inputs)
        for result in chain_results.get("result"):
            result["response"] = result["response"].strip()

        return self._build_response_dict(chain_results)

    # json_schema 格式
    def json_schema(self, json_schema: dict,
                    template: str,
                    input_variables: List[str] = None,
                    inputs: dict[str:str] = None, **kwargs) -> dict[str, Union[Optional[str], Any]]:

        self._validate_inputs(template, input_variables, inputs, **kwargs)

        if json_schema is None:
            raise ValueError("Missing required keyword argument 'json_schema'")

        chain = jsonschema_output_parser(output_schema=json_schema, llm=self.llm, prompt=self.prompt, n=self.n)

        chain_results = chain.run(self.inputs)

        return self._build_response_dict(chain_results)


magicChain = MagicChain()
