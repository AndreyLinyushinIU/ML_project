import asyncio
import logging
import uuid
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from aiogram import Dispatcher
from aiogram import types
from aiogram.dispatcher import FSMContext

from PIL import Image

from .states import UserStates
from .models import ModelsRegistry, Model
from .keyboards import ModelChoiceKeyboardMarkupFactory


logger = logging.getLogger('handlers')

process_pool_executor = ThreadPoolExecutor(max_workers=10)


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
We will send you the result in several seconds."""


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

    result_image_path = await _apply_style_transfer(model, content_image_uuid, style_image_uuid)
    with open(result_image_path, 'rb') as image:
        await message.reply_photo(image)


async def _apply_style_transfer(model: Model, content_image_uuid, style_image_uuid: str) -> str:
    content_image_path = f'/tmp/{content_image_uuid}'
    style_image_path = f'/tmp/{style_image_uuid}'
    result_image_path = f'/tmp/{uuid.uuid4().hex}.jpg'

    content_image = Image.open(content_image_path)
    style_image = Image.open(style_image_path)

    run_func = partial(model.run_and_save, content_image, style_image, result_image_path)
    fut = asyncio.get_running_loop().run_in_executor(process_pool_executor, run_func)
    await fut

    return result_image_path
