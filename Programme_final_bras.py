import cv2
import dlib
import numpy as np
from imutils import face_utils
from math import sqrt
from time import sleep
import RPi.GPIO as GPIO

DIR1 = 20   # Direction GPIO Pin
STEP1 = 21  # Step GPIO Pin
DIR2= 19
STEP2= 26
DIR3= 6
STEP3= 13
CW = 1     # Clockwise Rotation
CCW = 0    # Counterclockwise Rotation
SPR = 48   # Steps per Revolution (360 / 7.5)

#Configuration des GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR1, GPIO.OUT)
GPIO.setup(STEP1, GPIO.OUT)
GPIO.setup(DIR2, GPIO.OUT)
GPIO.setup(STEP2, GPIO.OUT)
GPIO.setup(DIR3, GPIO.OUT)
GPIO.setup(STEP3, GPIO.OUT)
pwm_gpio = 12
frequence = 50
GPIO.setup(pwm_gpio, GPIO.OUT)
pwm = GPIO.PWM(pwm_gpio, frequence)

step_count = SPR
delay = .0208
ratio=(12.5-4)/180
rot=ratio*20
list_eard=[]

class HeadPose():
    """Estimation of the angles of Euler"""
    
    def __init__(self):
        self.face_landmark_path = './shape_predictor_68_face_landmarks.dat'
        
    def get_point(self):
        """Get 2D coordonates, 3D correspondancy and camera parameters"""
        
        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        self.cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                 [1.330353, 7.122144, 6.903745],
                                 [-1.330353, 7.122144, 6.903745],
                                 [-6.825897, 6.760612, 4.402142],
                                 [5.311432, 5.485328, 3.987654],
                                 [1.789930, 5.393625, 4.413414],
                                 [-1.789930, 5.393625, 4.413414],
                                 [-5.311432, 5.485328, 3.987654],
                                 [2.005628, 1.409845, 6.165652],
                                 [-2.005628, 1.409845, 6.165652],
                                 [2.774015, -2.080775, 5.048531],
                                 [-2.774015, -2.080775, 5.048531],
                                 [0.000000, -3.116408, 6.097667],
                                 [0.000000, -7.415691, 4.070434]])

        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                   [10.0, 10.0, -10.0],
                                   [10.0, -10.0, -10.0],
                                   [10.0, -10.0, 10.0],
                                   [-10.0, 10.0, 10.0],
                                   [-10.0, 10.0, -10.0],
                                   [-10.0, -10.0, -10.0],
                                   [-10.0, -10.0, 10.0]])

        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        
    def get_pose(self,shape):
        """Get pose estimation from 2D image"""
        self.image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, self.image_pts,
                                                        self.cam_matrix,self.dist_coeffs)

        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec,
                                            self.cam_matrix,self.dist_coeffs)

        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        
        return reprojectdst, euler_angle
    
    def eye_aspect_ratio(self,shape):
         # compute the euclidean distances between the two sets of
         # vertical eye landmarks (x, y)-coordinates
    
       A = self.dist(shape[1], shape[5])
       B = self.dist(shape[2], shape[4])
       # compute the euclidean distance between the horizontal
       # eye landmark (x, y)-coordinates

       C = self.dist(shape[0], shape[3])
       # compute the eye aspect ratio
   
       ear=(A+B)/(2.0*C)
       # return the eye aspect ratio
       return ear

    def dist(self,A,B):
       dist_eucli = sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
       return dist_eucli


    def detect_blink(self,eard) :
        seuil=0.25
        if eard < seuil :
            print("clignement")
            return True
        
        else :
            #print("pas de clignement")
            return False   
    
    def run_pose_estimation(self):
        """Run camera capture"""
        
        self.get_point()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_landmark_path)

        #pos_servo vaut 0 si la pince est ouverte et 1 si elle est fermée
        pos_servo = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                face_rects = detector(frame, 0)

                if len(face_rects) > 0:
                    shape = predictor(frame, face_rects[0])
                    shape = face_utils.shape_to_np(shape)
                    
                    self.shape_right_eye=shape[36:42]

                    reprojectdst, euler_angle = self.get_pose(shape)

                    """for (x, y) in shape:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)"""
                        
                    for (x, y) in shape[36:48]  : 
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    
                    for start, end in self.line_pairs:
                        cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 255, 255))

                        cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)
                        cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
                        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)
                
                        eard= self.eye_aspect_ratio(self.shape_right_eye)
                        #earg=eye_aspect_ratio_g(shape_left_eye)
                        #print (eard,earg)
                        print (eard)
               
                        self.detect_blink(eard)
                
                
                        list_eard.append(self.eye_aspect_ratio(self.shape_right_eye))
                        
                        #DROITE
                        if euler_angle[1,0]<18:
                               i=0
                               for k in range(5):
                                   if euler_angle[1,0]<18:
                                       i+=1
                               if i>3:
                                   GPIO.output(DIR1, CW)
                                   #GPIO.output(DIR2, CW)
                                   #GPIO.output(DIR3, CW)
                                   for x in range(step_count):
                                       GPIO.output(STEP1, GPIO.HIGH)
                                       #GPIO.output(STEP2, GPIO.HIGH)
                                       #GPIO.output(STEP3, GPIO.HIGH)
                                       sleep(delay)
                                       GPIO.output(STEP1, GPIO.LOW)
                                       #GPIO.output(STEP2, GPIO.LOW)
                                       #GPIO.output(STEP3, GPIO.LOW)
                                       sleep(delay)
                                             
                        #GAUCHE
                        if euler_angle[1,0]>48:
                               i=0
                               for k in range(5):
                                   if euler_angle[1,0]>48:
                                       i+=1
                               if i>3:
                                   GPIO.output(DIR1, CCW)
                                   #GPIO.output(DIR2, CCW)
                                   #GPIO.output(DIR3, CCW)
                                   for x in range(step_count):
                                       GPIO.output(STEP1, GPIO.HIGH)
                                       #GPIO.output(STEP2, GPIO.HIGH)
                                       #GPIO.output(STEP3, GPIO.HIGH)
                                       sleep(delay)
                                       GPIO.output(STEP1, GPIO.LOW)
                                       #GPIO.output(STEP2, GPIO.LOW)
                                       #GPIO.output(STEP3, GPIO.LOW)
                                       sleep(delay)

                        #BAS          
                        if euler_angle[2,0]<-2:
                               i=0
                               for k in range(5):
                                   if euler_angle[1,0]>-2:
                                       i+=1
                               if i>3:
                                   GPIO.output(DIR2, CCW)
                                   #GPIO.output(DIR3, CCW)
                                   for x in range(step_count):
                                       GPIO.output(STEP2, GPIO.HIGH)
                                       #GPIO.output(STEP3, GPIO.HIGH)
                                       sleep(delay)
                                       GPIO.output(STEP2, GPIO.LOW)
                                       #GPIO.output(STEP3, GPIO.LOW)
                                       sleep(delay)
                        
                        #HAUT
                        if euler_angle[2,0]>3:
                            i=0
                               for k in range(5):
                                   if euler_angle[1,0]>3:
                                       i+=1
                               if i>3:
                                   GPIO.output(DIR2, CCW)
                                   #GPIO.output(DIR3, CCW)
                                   for x in range(step_count):
                                       GPIO.output(STEP2, GPIO.HIGH)
                                       #GPIO.output(STEP3, GPIO.HIGH)
                                       sleep(delay)
                                       GPIO.output(STEP2, GPIO.LOW)
                                       #GPIO.output(STEP3, GPIO.LOW)
                                       sleep(delay)
                                
                        #CLIGNEMENT
                        if self.detect_blink(eard) == True:
                            i=0
                               for k in range(5):
                                   if euler_angle[1,0]>3:
                                       i+=1
                                if i>3:
                                    if (pos_servo=1):
                                        pwm.start(4)
                                        pwm.ChangeDutyCycle(rot)
                                        pos_servo=0
                                        sleep(delay)
                                    else (pos_servo=0):
                                        pwm.start(100)
                                        pwm.ChangeDutyCycle(rot)
                                        sleep(delay)
                                        pos_servo=1

                        
                                
                        sleep(.5)




                cv2.imshow("demo", frame)



                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    pwm.stop()
                    GPIO.cleanup()
                    break



if __name__ == '__main__':
    head_pose_estimator = HeadPose()
    head_pose_estimator.run_pose_estimation()
