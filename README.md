# My Face Recognizer

#### A simple face recognition program built using Python, Face_Recognition library and OpenCV

<br>

The program is capable to do face recognition on Unknown pictures by learning faces from a Known set. The face_recognition library
is used for face encodings, finding the face locations in a picture - the model used for face detection on unknown set is Convolutional
Neural Network (cnn), and Comparison.

The project has two folders --
-  known_faces containing the faces used for training and 
-  unknown_faces containing the faces used for testing / evaluation

## Usage
```
pip install -r requirements.txt

python face_recognition_example.py

```

