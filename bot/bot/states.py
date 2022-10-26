from aiogram.dispatcher.filters.state import State, StatesGroup


class UserStates(StatesGroup):
    choosing_model = State()
    uploading_content_image = State()
    uploading_style_image = State()
