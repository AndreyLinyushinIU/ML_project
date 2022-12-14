import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from aiogram import Dispatcher
from aiogram import types
from aiogram.dispatcher import FSMContext

from .keyboards import ModelChoiceKeyboardMarkupFactory
from .models import ModelsRegistry, Model
from .states import UserStates

logger = logging.getLogger('handlers')

executors = {
    '1': ThreadPoolExecutor(max_workers=3),
    '2': ThreadPoolExecutor(max_workers=1),
    '3': ThreadPoolExecutor(max_workers=3)
}


def register_handlers(dp: Dispatcher, keyboard_markup_factory: ModelChoiceKeyboardMarkupFactory):
    dp.register_message_handler(start_command, commands=['start'])
    dp.register_message_handler(run_command, commands=['run'])
    dp.register_callback_query_handler(chose_model, keyboard_markup_factory.filter(), state=UserStates.choosing_model)
    dp.register_message_handler(uploaded_content_image, content_types=[types.ContentType.PHOTO], state=UserStates.uploading_content_image)
    dp.register_message_handler(uploaded_style_image, content_types=[types.ContentType.PHOTO], state=UserStates.uploading_style_image)


GREETINGS_MESSAGE = """
Hello! 👋 We are glad to see you in our Telegram bot.\n
We offer a couple of style transfer models.
To try one please type \\run command, choose the model, upload 2 images and then wait)
We will send you the result in several minutes."""


async def start_command(message: types.Message):
    await message.answer(GREETINGS_MESSAGE)


async def run_command(message: types.Message, state: FSMContext, keyboard_markup_factory: ModelChoiceKeyboardMarkupFactory):
    await state.set_state(UserStates.choosing_model)
    await message.answer('Choose the model', reply_markup=keyboard_markup_factory.new())


async def chose_model(callback_query: types.CallbackQuery, callback_data: dict[str, str], state: FSMContext):
    model_id = callback_data.get('model_id')
    if model_id is None:
        logger.error('no model_id in callback data')
        await state.finish()
        await callback_query.answer('Ooops something went wrong')
        return

    await callback_query.answer()
    await callback_query.message.edit_reply_markup()

    await state.set_state(UserStates.uploading_content_image)
    await state.update_data({'model_id': model_id})
    await callback_query.message.answer('Upload content image')


async def uploaded_content_image(message: types.Message, state: FSMContext):
    image_uuid = uuid.uuid4().hex
    await message.photo[-1].download(f'/tmp/{image_uuid}')
    await state.set_state(UserStates.uploading_style_image)
    await state.update_data({'content_image_uuid': image_uuid})
    await message.answer('Upload style image')


async def uploaded_style_image(message: types.Message, state: FSMContext, models_registry: ModelsRegistry):
    user_data = await state.get_data()
    await state.finish()

    content_image_uuid = user_data.get('content_image_uuid')
    if content_image_uuid is None:
        logger.error('no content_image_uuid in user data')
        await message.answer('Ooops something went wrong')
        return

    style_image_uuid = uuid.uuid4().hex
    await message.photo[-1].download(f'/tmp/{style_image_uuid}')

    model_id = user_data.get('model_id')
    if model_id is None:
        logger.error('no model_id in user data')
        await message.answer('Ooops something went wrong')
        return

    model = models_registry.get(model_id)
    if model is None:
        logger.error('invalid model_id in user data')
        await message.answer('Ooops something went wrong')
        return

    await message.answer(f'Estimated waiting time: {model.estimated_time_min} minutes')

    try:
        result_image_path = await _apply_style_transfer(model, content_image_uuid, style_image_uuid)
    except RuntimeError as e:
        logger.exception(e)
        await message.answer('Sorry, something went wrong')
        return

    with open(result_image_path, 'rb') as image:
        await message.reply_photo(image)


async def _apply_style_transfer(model: Model, content_image_uuid, style_image_uuid: str) -> str:
    content_image_path = f'/tmp/{content_image_uuid}'
    style_image_path = f'/tmp/{style_image_uuid}'
    result_image_path = f'/tmp/{uuid.uuid4().hex}.jpg'

    run_func = partial(model.run_and_save, content_image_path, style_image_path, result_image_path)
    executor = executors.get(model.id)
    if executor is None:
        raise RuntimeError(f'no executor for model {model.id}')
    fut = asyncio.get_running_loop().run_in_executor(executor, run_func)
    await fut

    return result_image_path
