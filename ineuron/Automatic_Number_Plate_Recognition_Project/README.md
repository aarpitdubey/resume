![img](./output/img1.png)

## Problem Statement

In the following project we will undestand to recognize the number plate automatically using Python, OpenCV and DeepLearning. Paddle OCR for character and digits extraction from car number plate.

Dataset Source : [Click Here to Download](https://storage.googleapis.com/kaggle-data-sets/686454/1203932/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240409%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240409T103317Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=b5dbd498bb697a74cf572611919800ab0dead9fe4044975588c916323605fbcb511711978102da2e1aab05cf071c62efe912e7a333b84e2a463b06d24bd172688270c72e1f7f2128a382891473293714952a390c7ffdd3ea4fb4ff45bba45863bf8ef9da374170bed0ad4c8acf55bf35c5d4f14779b97c9cc2bc514df3826ece014facb640c60173abe63d3cceb2d636a3f4696f88ed8d4d8e34d93251e5f235a6123946d0f4c5ed3c51c5230b476e8ca7eb2d953e0063f396e0a955830065205f66abde8a6f59d3b63b64ea3016e1c15f2456df3ba54e7a521daf7a275ac80f2ab12e5a379697a70e657c0860349bd3c37f1e9e94ea86bea4ee7ae56eb5483e "/")

https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

![img](./output/img2.png)

### Application Flow

This Application is used to detect the number plates from the image of Cars and the image is our data.

The Application flow is set on six different different stages:

1. Collect Required Data
2. Preprocess Data
3. Object Detection
4. Optical Character Recognition (OCR)
5. Web Application
6. Deployment

**Collect Required Data :** The first step, is to collect the required data, in my case for this project I collect the required data from open source platform Kaggle for datasets. So, from Kaggle I get the zip file with which contains all the data (images) with their annotations (annotations are basically the xml files which contains all the details about the images or data).

**Preprocess Data :** In the second step, I have to preprocess the required data. I had parse the information (Bounding box information) from the annotations (XML files)  and also then convert the annotation (XML) data to the dataframes. Here, I had un-structured data so after preprocessing and converting step that un-structured data convert into structured data.

**Object Detection :** In step third, I had applied object detection technique using the Deep learning, to extract the number plate from the car's image. This step is basically, used to give the relevant portion from the image automatically, so we can segmentized the part of image accordingly.

**Optical Character Recognition (OCR) :** In step four, I had applied optical character recognition (ocr) technique using paddle ocr because at this stage still we have portion of image data or relevant portion of image data but I want to extract the characters written over that image data and also the numeric character or digits from the image.

**Web Application :** In fifth step, I'm going to design and deploy the application in web environment using the flask framework and using some basic html and css basic web technologies. This step indicates i had deploy the application in the developmnt environment.

**Deployment :** In step six, at this step I had already create and deploy the object detection or deep learning application in Development environment but the actual deployment is creating a pipleline to the cloud deployment or production-ready environment this step indication the actual deployment in the Production environment. It is the work of DevOps or MLOps or AIOPs  Engineer. I select the AWS Cloud for this step and for this application.

### Application Flow Diagram

I had created this simple diagram to show the above steps.

![img](./output/app_flow.png)

### Use cases of Automatic Number Plate Detection:

Automatic Number Plate Recognition (ANPR) technology has various real-time use cases that provide significant benefits in different sectors. In today's world, security is becomming a major concern for most of the organisation, institutions, corporates and essentials services. In that sense the security is the primary concern or a priority for any Organisation for that possibily, in parking areas many times rules, laws were broken un-intentionally or intentionally. Here are some common use cases along with their explanations:

1. **Vehicle Parking :** ANPR is commonly used in vehicle parking management systems to automate entry and exit processes. When a vehicle enters a parking lot, ANPR cameras capture the vehicle's license plate, allowing for accurate and efficient tracking of parked vehicles. This technology can also help in managing parking space availability and enabling automated payment systems.
2. **Toll Enforcement :** ANPR is extensively used in toll enforcement systems to identify vehicles passing through toll booths without paying the required fees. By capturing license plate details, ANPR technology enables authorities to track and penalize offenders effectively.
3. **Vehicle Surveillance :** ANPR plays a crucial role in vehicle surveillance applications by monitoring and tracking vehicles in real-time. Law enforcement agencies use ANPR to identify stolen vehicles, track wanted vehicles, and enhance overall security. By integrating ANPR with surveillance systems, authorities can quickly locate and apprehend suspicious or unauthorized vehicles.
4. **Traffic Control :**  ANPR technology is widely used in traffic control scenarios to manage traffic flow, monitor congestion, and enforce traffic regulations. By capturing license plate data, ANPR systems can detect vehicles violating traffic laws, such as speeding or running red lights. This information enables authorities to take immediate action and improve overall traffic management.
5. **Access Control and Security:** ANPR is used for access control and security purposes in various facilities such as airports, corporate offices, and residential complexes. By automatically identifying vehicles entering or exiting premises, ANPR systems help enhance security measures and streamline access management processes.
6. **Law Enforcement and Investigations:** ANPR technology is instrumental in law enforcement and investigations by providing valuable data on vehicle movements and identifying suspects involved in criminal activities. Police agencies use ANPR to track suspect vehicles, locate missing persons, and gather evidence for investigations.
7. **Smart City Initiatives:** ANPR plays a critical role in smart city initiatives by facilitating efficient traffic management, enhancing public safety, and improving urban planning. By leveraging ANPR technology, cities can implement data-driven solutions for transportation optimization, emergency response coordination, and sustainable urban development.

![img](./output/usecases.png)

These use cases demonstrate the versatility and effectiveness of ANPR technology in various applications, showcasing its value in enhancing operational efficiency, improving security measures, and enabling data-driven decision-making.

## Project Implementation

Create a logger folder to get all the logs of our application

![img](./output/loggerfilecreated.gif "Author: Arpit Dubey")

And then, create a `__init__.py` and in same way we can create each folder manually and then put `__init__.py` file with some code.

Or their is another way creating a `template.py` file to proceed the same but in little automated way let's try the 2nd approach first because it is a simple approach and our work and client's work become easier.

Create a `template.py` file and type the below code to get the path

```python
import os
from pathlib import Path

random_path = 'google.com/ArpitDubey/detail.txt'

print(Path(random_path))
```

![img](./output/filePath.gif "Author: Arpit Dubey")

It also converts the forward slashes ("/") to the backward slashes ("\\")

![img](./output/filePath.png)

Now, you got it is a file or module which contains some python code within it.

Let's try to create a package with a name `APNR` and then create required folders and `__init__.py` files like:

1. src folder/`__init__.py` file
2. component folder/`__init__.py` file
3. config folder/`__init__.py` file
4. constants folder/`__init__.py` file
5. entity folder/`__init__.py` file
6. exception folder/`__init__.py` file
7. logger folder/`__init__.py` file
8. utils folder/`__init__.py` file
9. pipeline folder/`__init__.py` file
10. notebook folder/`data_collection.ipynb` and one more `ocr.ipynb`
11. `requirements.txt` file
12. app.py file

### Folder Structure Architecture

![img](./output/folderstructure.png "/")

Manual code code to create folder structure:

```python
import os
from pathlib import Path 

package_name = 'com.aarpitdubey'

list_of_files = [
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/component/__init__.py",
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
```

code implementation here :

![img](./output/code.gif "Author: Arpit Dubey")

but In component folder we want `data_ingestion.py`, `data_transformation.py`, `prepare_base_model.py`, `model_trainer.py` and `model_pusher.py` files too

```python
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
```

let's add then and then see the output result:

![img](./output/code_updated.gif "Author: Arpit Dubey")

**NOTE**: In gif i made a correction instead of `model_pusher.py` I wrote `model-pusher.py` I just corrected it manually.

Creating a `setup.py` file

```python
import setuptools # type: ignore

VERSION = '0.0.1'
PROJECT_NAME = 'Automatic Number Plate Recognition'
AUTHOR_NAME = 'Arpit Dubey'
DESCRIPTION = 'An application (or) a project which can recognize the number plates of cars'

setuptools.setup(
    version=VERSION,
    PROJECT_NAME=PROJECT_NAME,
    author=AUTHOR_NAME,
    description=DESCRIPTION,
    package_dir= {"":"src"},
    packages=setuptools.find_packages(where='src')
)

```

This `setup.py` file contains the information about the version, project name, author (who creates it), description (about the project or application functionalities) and packages used or where to install those packages.

![img](./output/setup.gif)

Now, we have to mentioned the packages we are using to make this project that will in our `requirements.txt` file

```plaintext
pandas
tensorflow==2.10
scikit-learn
opencv-python-headless
boto3==1.26.3
mypy-boto3-s3==1.24.76
from_root==1.0.2
python-dotenv
paddleocr
paddlepaddle
flask
numpy==1.23.5
-e .
```

![img](./output/requirements.gif "Author: Arpit Dubey")

Here to install all the written packages we have to first **creating our environment** for that:

`conda create -n <name_of_the_environment> python==<version>\`

In my case, I select the name of my vitual environment as `apnr` and python version `3.8`

`conda create -n anpr python==3.8`

![img](./output/creating_env.gif "Author: Arpit Dubey")

install all the packages mentioned in `requirements.txt`

Note: activate your virtual environment first.

`pip install -r requirements.txt`

![img](./output/installing_packages.gif)

Now we have install all the required packages

let's start working with `Jupyter Notebook` first

### Working with Jupyter Notebook

After Activating your environment

![img](./output/activatingenvironment.gif "Author: Arpit Dubey")

let's install the Ipykernel and jupyternote book

`> pip install ipykernel jupyter notebook`

then after it installed type this command 

 `> activate <env_name>`

`> jupyter notebook`

![img](./output/jupyter_notebook.gif "Author: Arpit Dubey")

I used VSCode editor with jupyter notebook extention it makes me feel easy to do my work!

Let's import some useful libraries in jupyter notebook

```python
# Import Libraries

import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
```

NOTE: We add `os` library to understand the path of current working directory.

But, check you current directory first:

`os.getcwd()`

and if you want to change it use this code:

`os.chdir('<your dataset path>')`

in my case I change it because my current directory for dataset is different 

`os.chdir('E:\\AnacondaWorkspace\\resume\\ineuron\\Automatic_Number_Plate_Recognition_Project\\Dataset')`

To extract data from the annotation file or extract data xml file we have to import some more useful libraries

```python
import xml.etree.ElementTree as xet
from glob import glob
```
