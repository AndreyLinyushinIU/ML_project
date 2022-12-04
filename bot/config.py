from configargparse import ArgumentParser, ArgumentDefaultsHelpFormatter, YAMLConfigFileParser


def setup_args_parser() -> ArgumentParser:
    parser = ArgumentParser(
        default_config_files=['config.yml'],
        config_file_parser_class=YAMLConfigFileParser,
        args_for_setting_config_path=['-c', '--config-file'],
        config_arg_help_message='Config file path',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    bot_group = parser.add_argument_group('bot')
    bot_group.add_argument('--bot-token', type=str, help='Telegram bot token (from @BotFather)')

    redis_group = parser.add_argument_group('redis')
    redis_group.add_argument('--redis-enabled', type=bool, default=False, help='Flag to enable Redis storage')
    redis_group.add_argument('--redis-ip', type=str, default='localhost', help='IP of redis server')
    redis_group.add_argument('--redis-port', type=int, default=6379, help='Port of redis server')
    redis_group.add_argument('--redis-db', type=int, default=1, help='Redis database number')

    return parser
