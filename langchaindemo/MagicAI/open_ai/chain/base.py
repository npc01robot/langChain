from typing import Any, Dict, Union, Type, Optional

from langchain import BasePromptTemplate
from langchain.chains.openai_functions.base import convert_to_openai_function, _get_openai_output_parser
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.schema import BaseLLMOutputParser
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel

from langchaindemo.MagicAI.custom import MagicLLMChain


def jsonschema_output_parser(
        output_schema: Union[Dict[str, Any], Type[BaseModel]],
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate,
        n: int,
        *,
        output_parser: Optional[BaseLLMOutputParser] = None,
        **kwargs: Any,
) -> MagicLLMChain:
    """
    the input schema
    example:
    json_schema = {
        "title": "Person",
        "description": "Identifying information about a person.",
        "type": "object",
        "properties": {
            "name": {"title": "Name", "description": "The person's name", "type": "string"},
            "age": {"title": "Age", "description": "The person's age", "type": "integer"},
            "score": {"title": "Score", "description": "The person's score", "type": "number"},
            "fav_food": {
                "title": "Fav Food",
                "description": "The person's favorite food",
                "type": "string",
            },
        },
        "required": ["name", "age", "score"],
    }

    """

    if isinstance(output_schema, dict):
        function: Any = {
            "name": "output_formatter",
            "description": (
                "Output formatter. Should always be used to format your response to the"
                " user."
            ),
            "parameters": output_schema,
        }
    else:

        class _OutputFormatter(BaseModel):
            """Output formatter. Should always be used to format your response to the user."""  # noqa: E501

            output: output_schema  # type: ignore

        function = _OutputFormatter
        output_parser = output_parser or PydanticAttrOutputFunctionsParser(
            pydantic_schema=_OutputFormatter, attr_name="output"
        )
    functions = [function]
    if not functions:
        raise ValueError("Need to pass in at least one function. Received zero.")
    openai_functions = [convert_to_openai_function(f) for f in functions]
    fn_names = [oai_fn["name"] for oai_fn in openai_functions]
    output_parser = output_parser or _get_openai_output_parser(functions, fn_names)
    llm_kwargs: Dict[str, Any] = {
        "functions": openai_functions,
    }
    if len(openai_functions) == 1:
        llm_kwargs["function_call"] = {"name": openai_functions[0]["name"]}
    star_chain = MagicLLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=output_parser,
        llm_kwargs=llm_kwargs,
        output_key="response",
        n=n,
        **kwargs,
    )
    return star_chain
