import os
from pathlib import Path

list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "src/logger/logging.py",
    "src/exception/exception.py",
    "test/unit/__init__.py",
    "test/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiment/experiments.ipynb"

]

for filepath in list_of_files:
    filepath = Path(filepath)
    file_dir, file = os.path.split(filepath) #split path and file name
    if file_dir != "": # if file_dir have path then create the following file in it
        os.makedirs(file_dir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(file) == 0):
        """
        if path does not exist or if my file is empty then open it. For that it needed to be created
        """
        with open(filepath, "w") as f:
            pass