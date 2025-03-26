# we will define our helper function in our utils file

import os 
from box.exceptions import BoxValueError
import yaml
from Heart_Segmentation import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import zipfile


# this ensure_annotations is a decorator
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file 

    Args:
        path_to_yaml(str): path like input
    
    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns: 
        ConfigBox: Configbox Type
    """

    try: 
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directions: list, verbose = True):
    """create list of directories

    Args:
        path_to_directions(list): list of path of directories
        ignore_log(bool , optional): ignore if multiple dirs is to be created. Defaults to False.
    """

    for path in path_to_directions:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations #save data of example confusion matrix i.e accuracy , loss value etc
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path(Path): path to the json file
        data(dict) : data to be saved in json file    
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at : {path}")


    # --------------extra functions

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path(Path): path to json file

    Returns:
        COnfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file is loaded succesfully from : {path}")
    return ConfigBox(content)
    

@ensure_annotations
def save_bin(data: Any , path : Path):
    """save binary file 

    Args:
        data(Any): data to be saved as binary
        path(Path): path to binary file
    """
    joblib.dump(value = data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path)-> Any:
    """load binary data
    
    Args:
        path(Path): path to binary file

    Returns:
        Any: object stored in the file
    """

    data = joblib.load(path)
    logger.info(f"binary file loaded from :{path}")
    return data

@ensure_annotations
def get_size(path: Path)-> str:
    """get the size in KB

    Args:
        path(Path) : path of the file

    Returns: 
        str: size in the KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"

