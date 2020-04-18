
import os
import requests
from zipfile import ZipFile
from loguru import logger
import tensorflow_datasets as tfds


URL = 'https://www.manythings.org/anki/fra-eng.zip'
FILENAME = 'data/fra-eng.zip'



def download_and_read_file(filename, url=None):
    if not os.path.exists(filename):
        session = requests.Session()
        response = session.get(url, stream=True)

        CHUNK_SIZE = 32768
        with open(filename, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    zipf = ZipFile(filename)
    filename = zipf.namelist()
    with zipf.open('fra.txt') as f:
        lines = f.read()

    return lines