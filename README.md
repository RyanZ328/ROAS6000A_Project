# ROAS6000A_Project_init

## keyboardcontrol node

Control linear or angle velocity
``` 
rosrun keyboard_control KeyBoardControl
```
Control linear and angle velocity at the same time
``` 
rosrun keyboard_control teleop.py
```
## Face Recognition node

Prerequisties:
``` 
pip install opencv-python numpy torch
```


Run Nodes:
``` 
rosrun face_recognition recognition.py
```
or
``` 
roslaunch face_recognition face_recognition.launch
```
