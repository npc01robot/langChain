from typing import List

from langchaindemo.MagicAI.open_ai import OpenAI
#  PalmAI
from langchaindemo.MagicAI.palm_ai import PalmAI

class Interface:
    def __init__(self):
        self.class_dict = {
            "OpenAI": OpenAI,
            "PalmAI": PalmAI
        }

    def create_factory(self, ai_name: str = None, **kwargs):
        class_obj = self.class_dict.get(ai_name)
        if class_obj:
            execute_instance = class_obj()
            return execute_instance
        else:
            raise ValueError(f"Invalid classname: {ai_name}")


class MagicAI:
    interface: Interface = Interface()

    def run(self,
            template: str,
            input_variables: List = None,
            inputs: dict = None,
            ai_name: str = "OpenAI",
            **kwargs):
        cls_run = self.interface.create_factory(ai_name)
        return cls_run.run(template, input_variables, inputs, **kwargs)

    def json_schema(self,
                    json_schema: dict,
                    template: str,
                    input_variables: List[str] = None,
                    inputs: dict[str:str] = None,
                    ai_name: str = "OpenAI",
                    **kwargs):
        cls_json_schema = self.interface.create_factory(ai_name)
        return cls_json_schema.json_schema(json_schema, template, input_variables, inputs, **kwargs)

    def chat(self, content, streaming, ai_name: str = "OpenAI", **kwargs):
        cls_chat = self.interface.create_factory(ai_name)
        return cls_chat.chat(content, streaming, **kwargs)

    def embeddings(self, input_data, ai_name: str = "OpenAI", **kwargs):
        cls_embed = self.interface.create_factory(ai_name)
        if isinstance(input_data, str):
            return cls_embed.embeddings(input_data, **kwargs)
        if isinstance(input_data, list):
            return cls_embed.embed_documents(input_data, **kwargs)
        raise TypeError("the Input Data must be string or list type,but you give a {}".format(type(input_data)))


magicAI = MagicAI()
