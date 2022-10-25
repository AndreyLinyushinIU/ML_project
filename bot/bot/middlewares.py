from typing import Any

from aiogram import types
from aiogram.dispatcher.middlewares import BaseMiddleware

from .models import ModelsRegistry
from .keyboards import ModelChoiceKeyboardMarkupFactory


class DIMiddleware(BaseMiddleware):

    def __init__(self, models_registry: ModelsRegistry, keyboard_markup_factory: ModelChoiceKeyboardMarkupFactory):
        super().__init__()
        self._models_registry = models_registry
        self._keyboard_markup_factory = keyboard_markup_factory

    async def _inject(self, data: dict[str, Any]):
        data['models_registry'] = self._models_registry
        data['keyboard_markup_factory'] = self._keyboard_markup_factory

    async def on_process_message(self, message: types.Message, data: dict[str, Any]):
        await self._inject(data)

    async def on_process_callback_query(self, callback_query: types.CallbackQuery, data: dict[str, Any]):
        await self._inject(data)
