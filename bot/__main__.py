import asyncio

from aiogram import Bot, Dispatcher
# from aiogram.contrib.fsm_storage.redis import RedisStorage2
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from models import Model1
from models import Model3
from DeepPhotoStyle_pytorch.model2 import Model2

from .bot.models import ModelsRegistry
from .bot.handlers import register_handlers
from .bot.middlewares import DIMiddleware
from .bot.keyboards import ModelChoiceKeyboardMarkupFactory
from .config import setup_args_parser
from .logger import setup_logger


async def main():
    args_parser = setup_args_parser()
    args = args_parser.parse_args()

    logger = setup_logger()

    bot = Bot(args.bot_token, parse_mode='HTML')
    # storage = RedisStorage2(host=args.redis_ip, port=args.redis_port, db=args.redis_db)
    storage = MemoryStorage()
    dp = Dispatcher(bot, storage=storage)

    model1 = Model1()
    logger.info('loading 1st model')
    model1.load()
    logger.info('finished loading 1st model')

    model2 = Model2()
    logger.info('loading 2nd model')
    model2.load()
    logger.info('finished loading 2nd model')

    model3 = Model3()
    logger.info('loading 3rd model')
    model2.load()
    logger.info('finished loading 3rd model')

    models_registry = ModelsRegistry()
    models_registry.register(model1)
    models_registry.register(model2)
    models_registry.register(model3)

    keyboard_markup_factory = ModelChoiceKeyboardMarkupFactory(models_registry)
    di_middleware = DIMiddleware(models_registry, keyboard_markup_factory)
    dp.setup_middleware(di_middleware)
    register_handlers(dp, keyboard_markup_factory)

    logger.info('starting bot')
    try:
        await dp.start_polling()
    finally:
        logger.info('stopping bot')
        session = await dp.bot.get_session()
        await session.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
