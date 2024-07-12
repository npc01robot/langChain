# langchainDemo

一个关于自定义langchain的Demo。

关于对langchain的魔改，主要用于chain embedding 流式输出等。
可以直接应用到django项目中

## 目录结构

```
├── MagicAI
│   ├── custom
│   │   ├── callbacks
│   │   │   ├── streamingweb.py  # 自定义的streamingweb回调函数,用于流式输出，统计token数量等
│   │   ├── llm_chain.py  # 自定义的langchain
│   ├── open_ai  # 针对 openai 的语言模型 进行构建chain embedding chat
│   │   ├── __init__.py  # openai 对外接口
│   │   ├── config.py  # 语言模型配置
│   │   ├── chain  # 语言模型chain
│   │   │   ├── base.py  # jsonschema 定义
│   │   │   ├── star_chain.py  # 语言模型chain
│   │   ├── chat  # 语言模型chain
│   │   │   ├── chat.py  # 语言模型chain
│   │   ├── embedding  # embedding 模型
│   │   │   ├── embed.py  # embedding 模型
│   ├── __init__.py  # 对外暴露的类接口，使用工厂模式进行实例化
│   ├── base.py  # 所有模型的基类
└── settings.py  # 项目配置
```

## 安装

```
pip install -r requirements.txt
```

## 运行
在config 文件夹中需要配置项目的settings.py文件

```
from django.conf import settings
from django.core.cache import cache
from common.proxy_pool import proxy_pool
```
基于 外层base 设计针对不同语言模型的包，加载进外层init文件中，并实例化。

```
        self.class_dict = {
            "OpenAI": OpenAI,
            "PalmAI": PalmAI
        }
```

对外使用run、json_schema、chat、embedding方法，具体使用方法参考各个方法的注释,查看基类以获取更多的传参案例。
```python
magicAI.run()
magicAI.json_schema()
magicAI.chat()
magicAI.embedding()
```
