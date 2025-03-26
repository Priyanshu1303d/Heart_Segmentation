import os
from pathlib import Path
import logging 

logging.basicConfig(level= logging.INFO, format = '[%(asctime)s]: %(message)s:')

project_name = 'Heart_Segmentation'

list_of_files = [
    "./github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    # Path is converting the strings "/" of list_of_files into '\' i.e of windowsPath

    # splitting the file name and file separately
    filedir , filename = os.path.split(filepath)

    if filedir != "":  # check if we get filedir and is not empty
        os.makedirs(filedir , exist_ok=True)
        # make dir of name filedir and check if its is already existing or not if not then create a new dir
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        # if the size of the file is 0 i.e its empty then 
        with open(filepath , "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        # already existing file is there 
        logging.info(f"{filename} already exists")

