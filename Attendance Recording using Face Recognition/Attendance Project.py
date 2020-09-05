import cv2
import numpy as np
import face_recognition
import os

# CREATING A LIST OF ALL THE NAMES (STEP1)

path = 'ImagesAttendance'  # Image folder
images = []                # Image List
classNames = []            # Name List
myList = os.listdir(path)  # Grabbing the names from the path
print(myList)
for cl in myList:          # Declaring myList as cl
    curImg = cv2.imread(f'{path}/{cl}')  # Reading the current images and names
    images.append(curImg)                # Appending curImg to images
    classNames.append(os.path.splitext(cl)[0])  # Appending the names to classNames. Split text is used in order to get only the first word
print(classNames)

# ENCODING THE IMAGES (STEP2)

def findEncodings(images):    # Defining an function called findEncodings
    encodeList = []           # List of encodings
    for img in images:        # calling images as img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # face recognition reads the img only in RGB, so converting BGR2RGB
        encode = face_recognition.face_encodings(img)[0]  # Encoding the given images
        encodeList.append(encode)       # Appending the encodings to encodeList
    return encodeList                   # After completing the encoding, return back to encodeList

encodeListknown = findEncodings(images)  # ENCODING THE IMAGES
print('Encoding Complete')

#FIND THE MATCHING IMG USING WEBCAM (STEP3)

cap= cv2.VideoCapture(0)  # Reading the webcam

while True:
    sucess, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # Resizing to get a faster results
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS) # Multiple images may be detected so giving face locations
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame) # Finding the encoding of webcam

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): # will grab the face from the current frame and encode the face and loop it together thats why we use zip
        matches = face_recognition.compare_faces(encodeListknown, encodeFace)
        faceDis =face_recognition.face_distance(encodeListknown, encodeFace)
        matchIndex = np.argmin(faceDis) # The lowest value will be the correct encoding
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1), (x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)







