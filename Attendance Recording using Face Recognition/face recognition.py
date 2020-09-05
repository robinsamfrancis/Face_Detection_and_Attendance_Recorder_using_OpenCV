import cv2
import numpy as np
import face_recognition

# Changing the images from BGR to RGB (STEP-1)

imgdhoni = face_recognition.load_image_file('ImagesBasic/dhoni.jpg')
imgdhoni = cv2.cvtColor(imgdhoni,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/virat.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# FINDING THE FACES AND ENCODINGS(STEP-2)
faceLoc = face_recognition.face_locations(imgdhoni)[0] # top, right, bottom, left
encodeDhoni = face_recognition.face_encodings(imgdhoni)[0]
cv2.rectangle(imgdhoni,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),3)

faceLocTest = face_recognition.face_locations(imgTest)[0] # top, right, bottom, left
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),3)

# COMPARING FACES AND COMPARING THE DISTANCE B/W THEM (STEP-3)

results = face_recognition.compare_faces([encodeDhoni],encodeTest)
faceDis = face_recognition.face_distance([encodeDhoni],encodeTest) # LOWER THE DISTANCE, BETTER THE MATCH
print(results,faceDis)
cv2.putText(imgTest,f'{results}{(faceDis,2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
cv2.imshow('Dhoni', imgdhoni)
cv2.imshow('Dhoni Test', imgTest)
cv2.waitKey(0)