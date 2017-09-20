import face_recognition
import cv2
import os
from os import listdir as ls


video_capture = cv2.VideoCapture(0)

files = [x for x in ls('./pics') if x[-4:]==(".jpg")]
name_list =[]
known_faces=[]
for file in files:
    Loaded_image=face_recognition.load_image_file('./pics/'+file)
    known_face_encoding = face_recognition.face_encodings(Loaded_image)[0]
    known_faces.append(known_face_encoding)
    #print(known_faces)
    name_list.append(os.path.splitext(file)[0])


face_locations = []
face_encodings = []
face_distances = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # match = face_recognition.compare_faces([my_face_encoding], face_encoding)
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distances = face_recognition.face_distance(known_faces,face_encoding)
            name = "Unknown"

            for i,match in enumerate(matches):
                if match and face_distances[i]<0.5:
                    name = name_list[i]
                    #print(i)

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if name =='Unknown':

            cv2.putText(frame,'Room is occupied with un authorized person',(100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2, cv2.LINE_AA)
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (200,0,200), 2)


        cv2.putText(frame,name,(right-130,right+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2, cv2.LINE_AA)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 250), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
