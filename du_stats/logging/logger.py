from datetime import datetime
import logging
import os

LOG_FILE=f"{datetime.now().strftime('%d_%m_%y_%H_%M_%S')}.log"
LOG_FILEPATH = os.path.join(os.getcwd(), 'logs', LOG_FILE)
LOG_DIR = os.path.dirname(LOG_FILEPATH)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILEPATH,
    format='[ %(asctime)s ] - %(filename)s - %(lineno)s - %(message)s',
    level=logging.INFO
)