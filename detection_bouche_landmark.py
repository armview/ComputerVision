# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:38:47 2020

@author: Solen
"""


import cv2
import dlib
import numpy as np
from imutils import face_utils
from math import sqrt
import matplotlib.pyplot as plt




                




             

def run():
    cap = cv2.VideoCapture(0) #indique le canal ur lequel sera r&cup l'image de la caméra

    if not cap.isOpened():
        print("Unable to connect to camera.")

    detector = dlib.get_frontal_face_detector()  #détecteur 
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #68 points du modèle déja défini

    
    


    while cap.isOpened() : #tant que ma cam exécute des images
        ret, frame = cap.read() #renvoie l'image frame et la valeur de retour en cas d'erreur
  
        if ret: 
            face_rects = detector(frame, 0)

            if len(face_rects) > 0: #si un visage est détécté alors prédiction pour trouver les différents points
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)
                
           
                
                for (x, y) in shape[48:60]  : 
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    
                

                cv2.imshow("demo", frame)
              
                
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    
                    
                    
                    cv2.destroyAllWindows()
                    break

                 
        
if __name__ == '__main__':
    run()