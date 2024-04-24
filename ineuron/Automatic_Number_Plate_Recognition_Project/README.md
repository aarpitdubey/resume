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

**Note**: In my case I change it because my current directory for dataset is different

`os.chdir('E:\\AnacondaWorkspace\\resume\\ineuron\\Automatic_Number_Plate_Recognition_Project\\Dataset')`

To extract data from the annotation file or extract data xml file we have to import some more useful libraries

```python
import xml.etree.ElementTree as xet
from glob import glob
```

![img](./output/working_xml_path.gif "Author: Arpit Dubey")

**Challenge 1: Current directory understanding problem**

**Challenge 2: Resolving with correct directory path**

**Challenge 3: Want both the image and xml annotations together.**

![img](./output/corrected_xml_path.gif "Author: Arpit Dubey")

Now, let's seperate the filepaths, xmin, xmax, ymin and ymax create a dictionary

![img](./output/working_xml_files.png "Author: Arpit Dubey")

**Step 1** : we have to parse the file

**Step 2** : find the root node for this xml tree

**Step 3**: find the first targeted tag then, we have to find the 2nd targeted tag

**Step 4:** Now, after finding the ` filenames, xmax, amin, ymax and ymin` we have to save them in a dictionary because then we convert it into a dataframe or csv file.

**Step 5:** As of now, we find the `xmax, xmin, ymax and ymin` but we don't have the image filename, find it and insert.

Now, before parsing we have to create a dictonary of lists so, we want to parse the annotations or xml files

for that we required five important things : filepath, xmin, xmax, ymin, ymax and these are in multiple values for different different images one xml file have xmin, xmax, ymin and ymax and there are multiples xml files and paths

```python
# Let's create a dictionary which have certain list as values, filepath=[], xmin[], xmax[], ymin[], ymax[]

label_dict = dict(filepath=[], xmin=[], xmax=[], ymin=[], ymax=[])

# There are multiple files and values so, we require a loop

for file in path:
    data = xet.parse(file) # parsing the files
    root = data.getroot() #getting the root
    object = root.find('object') # through root node we can find the object tag
    bnd_boxtag = object.find('bndbox') # through object tag we can find the bndbox (bounding box values)
  
    x_min = int(bnd_boxtag.find('xmin').text) # these 4 xmin, xmax, ymin, ymax are bounding box conditions
    x_max = int(bnd_boxtag.find('xmax').text)
    y_min = int(bnd_boxtag.find('ymin').text)
    y_max = int(bnd_boxtag.find('ymax').text)
  
    # Appending extracted coordinates into a dictonary
  
    label_dict['filepath'].append(file)
    label_dict['xmin'].append(x_min)
    label_dict['xmax'].append(x_max)
    label_dict['ymin'].append(y_min)
    label_dict['ymax'].append(y_max)

```

Now we have the all the 5 values of multiple files we have to create a DataFrame using pandas also generate a csv file for this information

##### Converted Extracted XML data into a dataframe

```python
df = pd.DataFrame(label_dict)
df.to_csv('labels.csv', index=False)
df.head()
```

![img](./output/convert_extracted_xml_data_into_dataframe.png "Author: Arpit Dubey")

![img](./output/find_the_img_path.png "Author: Arpit Dubey")

##### Extract respective image filename of XML and inserting it into a dataframe

```python
def get_filename(filename):
    image_filename = xet.parse(filename).getroot().find('filename').text
    image_filepath = os.path.join(os.getcwd(), 'common', image_filename)
    return image_filepath

image_path = list(df['filepath'].apply(get_filename))
```

![img](./output/find_the_img_path2.png "Author: Arpit Dubey")

**Here the execution of code:**

![img](./output/data_extraction_and_xml_parsing.gif "Author: Arpit Dubey")

Let's check out the **detection** using x_min, x_max, y_min and y_max coordinates in some random image.

```python
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from IPython.display import Image
from PIL import Image, ImageDraw
```

imported useful libraries and let's see a random image

```python
ORIGINAL_IMG = mpimg.imread(image_path[100]) # random image
plt.imshow(ORIGINAL_IMG)
```

```python
image_path[100]
```

**Output :**

    'e:\\AnacondaWorkspace\\resume\\ineuron\\Automatic_Number_Plate_Recognition_Project\\Dataset\\common\\Cars189.png'

**Execution of code :**

![img](./output/displaying_Original_image.gif "Author: Arpit Dubey")

**Displaying the detection on number plale and it's code:**

```python
# 1. Define the file path

image_path_ = image_path[100]
detected_number_plate = image_path_

# 2. Read the image using plotly

detected_number_plate_img = io.imread(detected_number_plate)

# 3. Define the coordinates;

x_min = 285
y_min = 241
x_max = 423
y_max = 286

# 4. Create the plotly figure and add the rectangle shape

fig = px.imshow(detected_number_plate_img)
fig.update_layout(width=800, height=500, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Fig. 1 - Cars189.png with bounding box')
fig.add_shape(type='rect', x0=x_min, x1=x_max, y0=y_min , y1=y_max, xref='x', yref='y', line_color='red')

# 5. Save the figure to an image file
image_filepath = os.path.join(os.getcwd(), 'Test', 'test_image.png')

# 6. Use the correct method to write the image
fig.write_image(image_filepath, engine='kaleido')

# 7. Display the saved image
img=mpimg.imread(image_filepath)
plt.imshow(img)
plt.axis('off')
plt.show()
```

**Execution of this code:**

![img](./output/detecting_number_plate_in_a_random_image.gif "Author: Arpit Dubey")

This is the result I'm getting: `bounding box coordinated well detected the number plates`

![img](./output/original_img_and_number_plate_detected_img.png)

#### Data Preprocessing

- Each image we will convert into an array using OpenCV
- Each image will be resize into (224, 224) because it is the standard compatible size of pre-trained transfer learning mdel i.e., Inceprion Resnet V2
- Normalizing the image by dividing with maximum number i.e., 255 (max. no. for 8-bit images) and the process is called normalization (Min-Max Scaler)
- Need to normalize the labels too because for Deep Learning model output range should be 0 to 1. For normalizing the labels I use to divide  the diagonal points with width and height of image.

**Coding :**

```python
# importing library

from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Creating some required variables and list

labels = df.iloc[:, 1:].values
output = []
data = []

# looping over each image

for index in range(len(image_path)):
  
    # image path
  
    image = image_path[index]
  
    # 1. Image to array
  
    image_array = cv2.imread(image)
    height, width, depth = image_array.shape


    # 2. Resizing image to (224, 224)
  
    load_image = load_img(image, target_size = (224, 224))
    load_image_array = img_to_array(load_image)


    # 3. Normalize image
  
    normalized_load_image_array = load_image_array/255.0


    # 4. Normalize labels
  
    xmin, xmax, ymin, ymax = labels[index]
    normalized_xmin, normalized_xmax = xmin/width, xmax/width
    normalized_ymin, normalized_ymax = ymin/height, ymax/height
    normalized_label = (normalized_xmin, normalized_xmax, normalized_ymin, normalized_ymax)

    # 5. Appending Outputs and Data
  
    data.append(normalized_load_image_array)
    output.append(normalized_label)
```

Spliting data into train and test split

```python
X = np.array(data, dtype = np.float32)
y = np.array(output, dtype = np.float32)

x_train, x_test, y_train, y_test = train_test_split(X, y,  train_size=0.8, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
```

**Execution of code:**

![img](./output/data_preprocessing.gif "Author: Arpit Dubey")

#### Preparing the Base Model

- A pre-trained model has been previously trained on a dataset and contains the weights and biases that represent the features of whichever dataset it was trained on.
- InceptionResNetV2, is a convolutional neural network that is trained on more than a million images from the ImageNet database.

**code:**

```python
# import Libraries

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input

# Neural Network Modeling

inception_renet = InceptionResNetV2(weights='imagenet', include_top=False,input_tensor=Input(shape=(224,224,3)))
output_model = inception_renet.output
output_model = Flatten()(output_model)
output_model = Dense(500, activation = 'relu')(output_model)
output_model = Dense(250, activation = 'relu')(output_model)
output_model = Dense(4, activation ='sigmoid')(output_model)

# Base Model 

model = Model(inputs = inception_renet.input, outputs= output_model)

# Compile our Model

model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=['accuracy'])

# Model Summary

model.summary()
```

This code is for building a neural network model using the InceptionResNetV2 architecture for image classification.

1. **Importing Libraries:**
   * The code starts by importing necessary libraries. These include functionalities from TensorFlow's Keras module.
2. **Neural Network Modeling:**
   * The InceptionResNetV2 model pre-trained on the ImageNet dataset is instantiated with its weights set to 'imagenet' and the top layers excluded.
   * The output of the InceptionResNetV2 model is then passed through a series of Dense layers, which are fully connected layers in a neural network. These Dense layers help in learning features from the extracted image features.
   * The last Dense layer has 4 neurons with sigmoid activation, indicating that this is a multi-label classification task.
3. **Base Model:**
   * The model is constructed using the input and output from the InceptionResNetV2 model.
4. **Compiling the Model:**
   * The model is compiled with a mean squared error (MSE) loss function, the Adam optimizer with a learning rate of 1e-4, and accuracy as the metric to monitor.
5. **Model Summary:**
   * Finally, the summary of the model is printed, showing the architecture of the model including the layers and parameters.

This code essentially **creates a custom neural network** by combining the powerful InceptionResNetV2 architecture with additional layers for specific task requirements, and then compiles the model for training and evaluation.

**Execution:**

![img](./output/base_model.gif "Author: Arpit Dubey")

#### Model Training

**Tensorboard**

- TensorBoard is a tool for providing the measurements and
  visualizations needed during the workflow.
- It enables tracking experiment metrics like loss and
  accuracy

**Batch Size**

- The batch size defines the number of samples that will be
  propagated through the network.
- let's say you have 1000 training samples and you want to
  set up a batch_size equal to 100. The algorithm takes the first 100 samples
  (from 1st to 100th) from the training dataset and trains the network.

```python



tensorboard = TensorBoard('object detection')



 



history = model.fit(x=x_train, y= y_train,batch_size=10,
epochs=10, validation_data (x_test,y_test),callbacks=[tensorboard])



 



# Saving our trained model



model.save('anpr_arpit_dubey_model.h5')



```

**Tensorboard**

- TensorBoard is a tool for providing the measurements and
  visualizations needed during the workflow.
- It enables tracking experiment metrics like loss and
  accuracy

**Batch Size**

- The batch size defines the number of samples that will be
  propagated through the network.
- let's say you have 1000 training samples and you want to
  set up a batch_size equal to 100. The algorithm takes the first 100 samples
  (from 1st to 100th) from the training dataset and trains the network.

```python
tensorboard = TensorBoard('object detection')

history = model.fit(x=x_train, y= y_train,batch_size=10,
epochs=10, validation_data (x_test,y_test),callbacks=[tensorboard])

# Saving our trained model

model.save('anpr_arpit_dubey_model.h5')
```

1. **TensorBoard Initialization:**

- `tensorboard = TensorBoard('object detection')`
  initializes TensorBoard with the provided name 'object detection'. TensorBoard
  is a visualization tool that comes with TensorFlow, used for monitoring and
  visualizing various aspects of the model training process.

2. **Model Training:**

   ```python
   model.fit(x=x_train, y=y_train, batch_size=10, epochs=10,
   validation_data=(x_test, y_test), callbacks=[tensorboard])
   ```

- `model.fit(x=x_train, y=y_train, batch_size=10, epochs=10, validation_data=(x_test, y_test), callbacks=[tensorboard])` trains the model using the training data `x_train` and `y_train`.
- The `batch_size=10` specifies how many samples are used in each training iteration.
- The `epochs=10` indicates the number of times the model will go through the entire training dataset.
- `validation_data=(x_test, y_test)` provides validation data to evaluate the model performance during training.
- `callbacks=[tensorboard]` includes the `tensorboard` callback to monitor the training process using TensorBoard.

3. **Model Saving:**

   - `model.save('anpr_arpit_dubey_model.h5')` saves the trained model to a file named 'anpr_arpit_dubey_model.h5'. This file can be later loaded to make predictions without retraining the model.

   ![img](./output/model_training.png "Author: Arpit Dubey")

### Functionality:

- The code segment initializes TensorBoard for visualization during model training, trains the model on the provided data, incorporates TensorBoard for monitoring, and finally saves the trained model for future use.

  **Execution:**

![img](./output/model_training.gif "Author: Arpit Dubey")
