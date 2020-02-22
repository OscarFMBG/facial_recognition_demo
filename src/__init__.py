import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
load_dotenv(Path(f"./{os.getenv('SECRET_ENV_FILE')}").resolve())
