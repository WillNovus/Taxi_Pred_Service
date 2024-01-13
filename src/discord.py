import os

from discordwebhook import Discord

from src.logger import get_logger

logger = get_logger()

def send_message_to_channel(message: str) -> None:

    try:
        DISCORD_WEBHOOK_URL = os.environ['DISCORD_WEBHOOK_URL']
    except KeyError:
        logger.warning('DISCORD_WEBHOOK_URL not found in environment variables. Skipping Discord notification.')
        return
    
    try:
        discord = Discord(url=DISCORD_WEBHOOK_URL)
        discord.post(content=message)
    except Exception as e:
        logger.error(f'Failed to send message to Discord Channel. Error: {e}')
        return