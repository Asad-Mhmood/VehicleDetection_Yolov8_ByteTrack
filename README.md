
# Vehicle Detection system
## Yolov8+ByteTrack

This project detect the Vehicle and count them. Yolov8 is used to detect the Vehicles and ByteTrack is used to track the vehicles.  
To build this project I use pretrained model which is trained on COCO dataset and my all classes which i need present in COCO dataset. So i do not train it on my customized data.  
But issue is that, I am not able to detect Auto. And this model consider Auto as a truck wihich is not correct.



## Demo
![Image](https://github.com/Asad-Mhmood/VehicleDetectionSystem/blob/master/data/1.PNG)
## Video
[Watch this video](https://youtu.be/MPkKICM0KFk?si=sRhvAUdfzI8AhFVi)




## Installation



```bash
  pip install -r requirements.txt
```
if Libraries/Dependencies are not install using this command than you have to download these libraries manualy using pip.
You have to go terminal and install each library using following command:
```bash
  pip install library-name
```

## How to use the Project
1: You have to download all dependencies which mention in requirements file. You can install these dependencies by the methods which mention in Installation section.  

2: After installation you have to add video in the project. You hav to give the path of the video in run.py where i mention. 

3: After this how have to change the limits of your line accordingly 

4: Chage the logic of detection according to your requirements. I mentionin the code where you have to make changes.

## Limitations

1: I have no class of auto in my dataset because i use pretrained model of yolov8. So for Auto it give false detection and it detect auto as truck and sometime detect as car.  


## Solution
To solve this issue To solve this issue train the model on customized data set which include include all classes.