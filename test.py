import os
from dotenv import load_dotenv, find_dotenv

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables*
load_dotenv(dotenv_path)

PATH_DATA_RAW = Path(os.getenv("PATH_DATA_RAW"))
GCP_KEY = Path(os.getenv("GCP_KEY"))
