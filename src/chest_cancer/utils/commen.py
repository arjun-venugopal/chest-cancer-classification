import os
from box.exceptions import BoxValueError
import yaml
from chest_cancer import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file to be read.

    Returns:
        ConfigBox: A ConfigBox object containing the contents of the YAML file.

    Raises:
        BoxValueError: If there is an error reading the YAML file.
    """

    try:
        with open(path_to_yaml, 'r') as stream:
            content = yaml.safe_load(stream)
            logger.info(f"Read yaml file from {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        logger.error(f"Error reading yaml file from {path_to_yaml}")
        raise BoxValueError
    except Exception as e:
        return e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates directories if they don't exist.

    Args:
        path_to_directories (list): A list of paths to the directories to be created.
        verbose (bool, optional): Whether to print verbose messages. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a JSON file.

    Args:
        path (Path): The path to the file to be saved.
        data (dict): The dictionary to be saved as a JSON file.

    Returns:
        None
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Saved json file at {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its contents as a ConfigBox object.

    Args:
        path (Path): The path to the JSON file to be loaded.

    Returns:
        ConfigBox: A ConfigBox object containing the contents of the JSON file.
    """
    with open(path, "r") as f:
        content = json.load(f)

    logger.info(f"Loaded json file from {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path): 
    """
    Saves a binary file using joblib.

    Args:
        data (Any): The data to be saved as a binary file.
        path (Path): The path to the file to be saved.

    Returns:
        None
    """
    joblib.dump(value= data, filename=path)
    logger.info(f"Saved binary file at {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads a binary file saved using joblib.

    Args:
        path (Path): The path to the file to be loaded.

    Returns:
        Any: The loaded data.
    """
    data = joblib.load(path)
    logger.info(f"Loaded binary file from {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName)-> str:
    """
    Decodes a base64 encoded string into an image and saves it to disk.

    Args:
        imgstring (str): The base64 encoded string.
        fileName (str): The path to the file to be saved.

    Returns:
        None
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    """
    Encodes an image file into a base64 encoded string.

    Args:
        croppedImagePath (str): The path to the image file to be encoded.

    Returns:
        bytes: The base64 encoded string of the image.
    """

    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    

    