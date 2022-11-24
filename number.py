import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import easyocr


import pandas as pd
import os.path
from local_utils import detect_lp
from tensorflow.keras.models  import model_from_json
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from os.path import splitext,basename
from PIL import Image, ImageChops
import math
from scipy import spatial

import streamlit as st 
import requests
import io

# ------------------------- FACE DETECTION MASK On&Off and returns all faces found
def doMaskOnOffDetection(opencv_image):
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    model = load_keras_model("mask_detector.model")

    image = opencv_image
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    
    
    # if not detections:
    #     st.text("No Detections were found")
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            percentage = doDatabaseIDMapping(face)
            # st.text("similarity percentage:{}".format(percentage))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            if percentage >0.95:
                color = (0, 255, 0) 
            else:
                color = (0,0,255)
                        
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    return opencv_image
    
def getFace(opencv_image):
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    model = load_keras_model("mask_detector.model")

    image = opencv_image
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    
    facesPresemt =[]
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            facesPresemt.append(face)

    return facesPresemt
    

def doDatabaseIDMapping(faceDetected):
    database_person = cv2.imread('male-worker.jpg')
    database_face = getFace(database_person)
    faceDetected = cv2.cvtColor(faceDetected, cv2.COLOR_BGR2RGB)
    
    face1 = np.array(database_face)
    face1 = face1.flatten()
    face1 = face1/255

        
    face2 = np.array(faceDetected)
    face2 = face2.flatten()
    face2 = face2/255
    
    similarity = -1 * (spatial.distance.cosine(face1, face2) - 1)
    
    return similarity 
        
        
            
# ----------------------------------------------------------------


#---------------------------- Number Plate Detection ------------------------------------
# Number Plate Detection with Haar Cascade 
def doNumberPlateDetectionCascade(img):
    # Load the cascade
    carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    carplate_rects = carplate_haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in carplate_rects: 
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) 
    return img


#Grab number plate from image using easyocr
def doANPR(img):

    # Convert int to uint -> np.array
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = np.array(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(gray)

    #Perform OCR
    reader = easyocr.Reader(['en']) 
        
    result = reader.readtext(gray)
    numberPlate = ""
    for item in range(len(result)):
        numberPlate+=result[item][1]
        numberPlate+=" "
    st.text("Number Plate Digits :{}".format(numberPlate))

   
#-------------------------CROP IMAGE TESTING---------------------------------------

# Update ANPR using Numberplatereco
def numberplateRecognizer(image):
    original_image = image
    #conver the numpy array into an Image type object
    h , w , c = image.shape
    image = np.reshape(image,(h,w,c))
    image = Image.fromarray(image, 'RGB')

    #convert image to bytes as api requests are in that format
    buf = io.BytesIO()
    image.save(buf,format = 'JPEG')
    byte_im = buf.getvalue()
    
    response = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        # data=dict(regions='hk'),  # Optional
        files=dict(upload=byte_im),
        headers={'Authorization': 'Token f184ad4cc54df68357e7873baea13a1d596ad6e4'}
    )
    json_response = response.json()
    # for key, value in json_response.items():
    #     st.write(key, "->", value)
    
    #Find number of number plate present in the image
    number_plates = len(json_response['results'])
    st.subheader("Total Number Plates Detected in this image:{}".format(number_plates))

    st.subheader("Number Plate Information")
    # get number plate and borders from the image
    for plate in json_response['results']:
        #Get numberplate and borders
        car_plate = plate['plate']
        xmin , ymin , xmax , ymax = plate['box'].values()
        
        #crop number plate and show borders
        numberplate_crop = image.crop((xmin , ymin , xmax , ymax))
        st.image(numberplate_crop)
        st.write("Number Plate Digits: {}".format(car_plate))
        
        # Draw Borders and show numberplate on image
        cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax),(255,0,0),5)
        cv2.putText(original_image , car_plate ,(xmin , ymin),fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0))
        
        
    original_image= cv2.cvtColor(original_image ,cv2.COLOR_BGR2RGB )
    # display number above number plate
    st.image(original_image)

#----------------------------------------------------------------------------------

def main():
    
    st.title("Open-CV Deep Learning Face Detection and Number Plate extraction")
    st.subheader("This program helps you detect people faces and perform Number Plate extraction")
    
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")
        
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original =  cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.imdecode(file_bytes, 1)
        display = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        st.image(display)
        
        # with st.sidebar:
        #     st.button("Do Face Detection and Face Matching")
        #     st.button("Do NumberPlate Extraction")
    
        if st.button("Do Face Detection and Face Matching"):
            #Face Detection with and without Mask 
            image = doMaskOnOffDetection(display)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.subheader("Face Detection and matching Algorithm")
            st.text("People who are present in the database will be highlighted in green")
            st.image(image)
        
        if st.button("Do NumberPlate Extraction"):
            numberplateRecognizer(original)

                    
        
        
        
        

if  __name__ == "__main__":
    main()
