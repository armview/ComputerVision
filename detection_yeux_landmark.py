import cv2
import dlib
import numpy as np
from imutils import face_utils
from math import sqrt
import matplotlib.pyplot as plt

list_eard=[]


                
"""
def eye_aspect_ratio_d(shape):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist(shape[37], shape[41])
    B = dist(shape[38], shape[40])
   
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist(shape[36], shape[39])
  
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    # return the eye aspect ratio
    return ear"""

def eye_aspect_ratio(shape):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    
    A = dist(shape[1], shape[5])
    B = dist(shape[2], shape[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates

    C = dist(shape[0], shape[3])
    # compute the eye aspect ratio
   
    ear=(A+B)/(2.0*C)
    # return the eye aspect ratio
    return ear

def dist(A,B):
    dist_eucli = sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    return dist_eucli

"""def detect_blink(eard,earg) :
    seuil=0.25
    if eard < seuil and earg < seuil :
        print("clignement")
        return True
    else :
        #print("pas de clignement")
        return False"""
    
def detect_blink(eard) :
    seuil=0.25
    if eard < seuil :
        print("clignement")
        return True
    else :
        #print("pas de clignement")
        return False              

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
                
                shape_right_eye=shape[36:42]
                shape_left_eye=shape[42:48]
                
                
                
                
                
                
                eard=eye_aspect_ratio(shape_right_eye)
                #earg=eye_aspect_ratio_g(shape_left_eye)
                #print (eard,earg)
                print (eard)
             
                detect_blink(eard)
                
                
                list_eard.append(eye_aspect_ratio(shape_right_eye))
                
                
                      
                
                     
                
                for (x, y) in shape[36:48]  : 
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    
                

                cv2.imshow("demo", frame)
              
                
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    
                    
                    
                    cv2.destroyAllWindows()
                    break

                 
        
if __name__ == '__main__':
    run()
    
plt.plot(list_eard)
plt.xlim(0,20)
plt.ylim(0,1)
plt.xlabel("temps")
plt.ylabel("distance EAR")

plt.show()

 

 
