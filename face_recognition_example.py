'''
Program - My Face Recognizer
Developer - Joydeep Banerjee
Description - The program is capable to do face recognition on Unknown pictures 
              by learning faces from a Known set. The face_recognition library
              is used for face encodings, finding the face locations in a picture - 
              the model used for face detection on unknown set is Convolutional
              Neural Network (cnn), and Comparison.
 '''

import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
# Higher less false positives
TOLERANCE = 0.6

FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'
known_faces = []
unknown_faces = []
known_names = []

print(" Fetching known faces")

#Looping through the files in known faces folder
for subdir in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{subdir}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{subdir}/{filename}")
        joydeep_encoding = face_recognition.face_encodings(image)[0]  # Encodes pictures into indexes
        known_faces.append(joydeep_encoding)
        known_names.append(subdir) # Here folder names would become rectangle labels

    
print(" Fetching unknown faces")

#Looping through the files in known faces folder
for subdir in os.listdir(UNKNOWN_FACES_DIR):
    for filename in os.listdir(f"{UNKNOWN_FACES_DIR}/{subdir}"):
        image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{subdir}/{filename}")
        locations = face_recognition.face_locations(image, model= MODEL) # recognize the faces (cordinates) in the pictures
        encodings = face_recognition.face_encodings(image, locations)
        # Preparing the image to be usable by OpenCV - 
        # converts an image from one color space to another
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 



        for face_encoding, face_location in zip(encodings, locations):
            # Compares the Known and the Unknown lists and returns a list of booleans( True / False)
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE) 
            match = None

            if True in results:
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')
                # This will put a rectangle on the detected face 
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                color = [255, 0, 0]
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # This is for the label under the top rectangle 
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)


        # Show image
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)
