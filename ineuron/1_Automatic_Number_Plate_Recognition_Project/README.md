![img](./output/img1.png)

## Problem Statement

In the following project we will undestand to recognize the number plate automatically using Python, OpenCV and DeepLearning. Paddle OCR for character and digits extraction from car number plate.

![img](./output/img2.png)

## Application Flow

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

**Web Application :** In fifth step, I'm going to design and deploy the application in web environment using the flask framework and using some basic html and css basic web technologies. This step indicates i had deploy the application in the development environment.

**Deployment :** In step six, at this step I had already create and deploy the object detection or deep learning application in Development environment but the actual deployment is creating a pipleline to the cloud deployment or production-ready environment this step indication the actual deployment in the Production environment. It is the work of DevOps or MLOps or AIOPs  Engineer. I select the AWS Cloud for this step and for this application.

### Application Flow Diagram

I had created this simple diagram to show the above steps.

![img](./output/app_flow.png)
