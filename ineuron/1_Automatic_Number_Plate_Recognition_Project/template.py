# Create by Arpit Dubey (aarpitdubey@gmail.com)
import os
from pathlib import Path 

package_name = 'com.aarpitdubey'

list_of_files = [
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/component/__init__.py",
    f"src/{package_name}/component/data_ingestion.py",
    f"src/{package_name}/component/data_transformation.py",
    f"src/{package_name}/component/prepare_base_model.py",
    f"src/{package_name}/component/model_trainer.py",
    f"src/{package_name}/component/model_pusher.py",
    f"src/{package_name}/config/__init__.py",
    f"src/{package_name}/constants/__init__.py",
    f"src/{package_name}/entity/__init__.py",
    f"src/{package_name}/exception/__init__.py",
    f"src/{package_name}/logger/__init__.py",
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/pipeline/__init__.py",
    "notebook/data_collection.ipynb",
    "notebook/ocr.ipynb",
    "requirements.txt",
    "app.py"
    
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    
    if file_dir != '':
        os.makedirs(file_dir, exist_ok=True)
        
    if(not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path, "w") as f:
            pass
