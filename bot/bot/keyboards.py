from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.callback_data import CallbackData

from .models import ModelsRegistry


class ModelChoiceKeyboardMarkupFactory:
    """
    Factory to create keyboard markups for model choice
    """

    def __init__(self, models_registry: ModelsRegistry):
        self._callback_data_factory = CallbackData('mid', 'model_id')
        self._models_registry = models_registry

    def new(self) -> InlineKeyboardMarkup:
        markup = InlineKeyboardMarkup(row_width=1)

        for model in self._models_registry:
            button = InlineKeyboardButton(
                text=model.name,
                callback_data=self._callback_data_factory.new(model_id=model.id)
            )
            markup.insert(button)

        return markup

    def filter(self):
        return self._callback_data_factory.filter()
