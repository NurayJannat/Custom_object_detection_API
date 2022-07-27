# CUSTOM OBJECT DETECTION API
This repository contains object detection api for detecting alchohol bottle. 
#### Dataset
https://www.kaggle.com/datasets/dataclusterlabs/alcohol-bottle-images-glass-bottles?select=1597906495385.jpg
#### Installation Process
Need to install few dependencies.

 1. First we need to install anaconda
Please follow this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart) to install anaconda
Use this link to download anaconda https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
 2. create a virtual env with the following command
	 ```
	  conda create -n env_name python=3.8.5
	  conda activate env_name
	```
	 
 3. Install other dependencies by using the following command
	 ``` 
	 cd path_to_project_folder
	 pip3 install -r requirements.txt
	 ```
3. clone tensorflow object detection api:
	``` 
	 git clone --depth 1 https://github.com/tensorflow/models
	```
3. Download "checkpoints" folder for object detection api from:
	``` 
	 https://drive.google.com/drive/folders/1IScb4XRMbwNLFi_m6Gj22gLD8WQXgUg6?usp=sharing
	```
 4. Setting up other configuration:
	 Go to this path: 
	 ``` 
	 cd models/research/
	 ``` 
	 Run this command:
	 ``` 
	 protoc object_detection/protos/*.proto --python_out=.
	 ``` 
	 If 'protoc' is not found or any error occurs regarding 'protoc', follow this http://google.github.io/proto-lens/installing-protoc.html tutorial (linux portion). Then run the command again. 

	 Otherwise jump to next step.
	 Run these commands next:
	``` 
	cp object_detection/packages/tf2/setup.py .
    python -m pip install .
	``` 
 6. Run the API:
    Make sure, present directory is project directory and run this command.

    ```
    python3 main.py
	```
    #### Test route 
    ```
    http://0.0.0.0:8000/object-detection
    ```
    ##### Request Body:
    {
        "img_base64": <base64 code of image>
    }
    ##### Response Body:
    {
        "output_image": <base64 code of image,
        "object_cords": [
        ]
    }
    ```
    http://0.0.0.0:8000/post_proc
    ```
    ##### Request Body:
    {
        "img_base64": <base64 code of image>,
        "x1": int 
        "y1": int
        "x2": int
        "y2": int
    }
    ##### Response Body:
    {
        "output_image": <base64 code of image,
    }
