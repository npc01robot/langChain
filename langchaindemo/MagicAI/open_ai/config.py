# import traceback
from django.conf import settings
from django.core.cache import cache
from common.proxy_pool import proxy_pool
OPEN_AI_KEY = settings.OPEN_AI_KEY

proxy_pool = proxy_pool()
cache = cache
class LLMConfig:
    api_key: str = OPEN_AI_KEY
    """open api key"""
    platform: str = settings.PLATFORM
    """Service running platform"""
    model: str = "gpt-3.5-turbo"
    """specify the chatgpt model"""
    book_key: str = "openai_proxy_book"
    openai_proxy: str = None

    def __init__(self):
        if self.platform == 'NT':
            proxy = cache.get_or_set(self.book_key, proxy_pool.random_proxy(index=0), 300)
            self.openai_proxy = f'http://{proxy}'
            print(self.openai_proxy)

llmConfig = LLMConfig()
