import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import easyocr

from IPython.display import Image


import argparse
import sys
import pandas as pd
import os.path
from local_utils import detect_lp
from tensorflow.keras.models  import model_from_json
from os.path import splitext,basename
from PIL import Image


import streamlit as st 




# Face Detection using Haar Cascade 
def doFaceDetection(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return img

# Number Plate Detection with Haar Cascade 
def doNumberPlateDetectionCascade(img):
    # Load the cascade
    carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    carplate_rects = carplate_haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in carplate_rects: 
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) 
    return img
    # cv2.imshow('NumberPlate Detection', img)
    # cv2.waitKey()
    

def doANPR(img):
    
    # Convert int to uint -> np.array
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = np.array(img)

    #Perform OCR
    reader = easyocr.Reader(['ch_sim','en']) 
        
    result = reader.readtext(img)
    st.subheader("Number Plate Digits :{}".format(result[0][1]))

    
    


    


#-------------------------CROP IMAGE TESTING---------------------------------------

## Method to load Keras model weight and structure files
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Model Loaded successfully...")
        return model
    except Exception as e:
        print(e)


#Extract NumberPlate Detected Region asImage
def get_plate(image_path, Dmax=608, Dmin = 608):
    
    image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image_path = image_path/255
    
    wpod_net_path = "models/wpod-net.json"
    wpod_net = load_model(wpod_net_path)
    
    vehicle = image_path
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    
    return LpImg

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
        
        #Perform Face Detection 
        image = doFaceDetection(opencv_image)
        
        # #Do Number Plate Detection 
        detections = doNumberPlateDetectionCascade(image)

        # Now do something with the image! For example, let's display it:
        st.image(detections, channels="BGR")
        # st.image(opencv_image)
        
        
        #Crop Number Plate Function Call 
        LpImg = get_plate(original)
        if LpImg:
            st.subheader("Number Plate Detected in the image:")
            croppedImage = LpImg[0]  
            
            st.image(croppedImage, channels="RGB")
            
            st.subheader("Character Recognition:")
            doANPR(croppedImage)
                
        
        
        
        
        
        
        
    
    
 
    # # # # cv2.imshow('detections:',image)
    
    # #Crop Number Plate Function Call 
    # img_name = cv2.imread('test-face.jpg')
    # # cv2.imshow('image',img_name)
    # # cv2.waitKey()
    # LpImg = get_plate(img_name)
    # croppedImage = LpImg[0]
    # doANPR(croppedImage)

    

    

    

if  __name__ == "__main__":
    main()
