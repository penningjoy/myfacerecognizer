import face_recognition
import os
import cv2

known_faces_dir = "known_faces"
unknown_faces_dir = "unknown_faces"
# Higher less false positives
Tolerence = 0.6

frame_thickness = 3
font_thickness = 2
model = 'cnn'
known_faces = []
unknown_faces = []
known_names = []

print(" Fetching known faces")

#Looping through the files in known faces folder
for subdir in os.listdir(known_faces_dir):
    for filename in os.listdir(f"{known_faces_dir}/{subdir}"):
        image = face_recognition.load_image_file(f"{known_faces_dir}/{subdir}/{filename}")
        joydeep_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(joydeep_encoding)
        known_names.append(subdir) # Here folder names would become rectangle labels

    
print(" Fetching unknown faces")

#Looping through the files in known faces folder
for subdir in os.listdir(unknown_faces_dir):
    for filename in os.listdir(f"{unknown_faces_dir}/{subdir}"):
        image = face_recognition.load_image_file(f"{unknown_faces_dir}/{subdir}/{filename}")
        locations = face_recognition.face_locations(image, model= model)
        encodings = face_recognition.face_encodings(image, locations)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



for face_encoding, face_location in zip(encodings, locations):
    results = face_recognition.compare_faces(known_faces, face_encoding, Tolerence)
    match = None

    if True in results:
        match = known_names[results.index(True)]
        print(f' - {match} from {results}')
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        color = (255, 0, 0)
        cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), font_thickness)


# Show image
cv2.imshow(filename, image)
cv2.waitKey(0)
cv2.destroyWindow(filename)





