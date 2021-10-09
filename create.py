import os
import cv2
import numpy as np
import pandas as pd
root = 'asl_alphabet_train'
import mediapipe as mp
import tensorflow as tf
import math

pi=math.pi
pi2=2*pi
def mcalc(m1,m2):
    a = m1 - m2
    if a>pi:
        a -= pi2
    elif a<-pi:
        a += pi2
    return a
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
refslope=[]
dist0=[]
dist2=[]
dist3=[]
dist4=[]
slope0=[]
slope2=[]
slope3=[]
slope4=[]
label=[]
for directory, subdirectories, files in os.walk(root):
# go through each file in that directory
    print(directory)
    #implemented
    # if(directory[-1]=='B'):
    #     break
    itr=0
    dire=directory.split('\\')
    dire.append('i')
    dire=dire[1]
    if(dire=='nothing'):
        continue
    for file in files:
    # read the image file and extract its pixels
        # print(file)
        frame = cv2.imread(os.path.join(directory,file))
        x , y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                # mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        # cv2.imshow("Output", frame)
            landmarks.pop(1)
            landmarks=np.array(landmarks)
            dist=[]  
            slope=[]
            for i in range(0,5):
                x=i*4
                li=[np.linalg.norm(landmarks[x]-landmarks[x+1]),np.linalg.norm(landmarks[x]-landmarks[x+2]),np.linalg.norm(landmarks[x]-landmarks[x+3])]
                ind=li.index(max(li))+1
                dist.append(li[ind-1])
                slope.append(math.atan2(landmarks[x+ind][1]-landmarks[x][1],landmarks[x+ind][0]-landmarks[x][0]))
            if(dist[1]==0):
                dist=[i+1 for i in dist]
            distratio=[dist[0]/dist[1],dist[1]/dist[1],dist[2]/dist[1],dist[3]/dist[1],dist[4]/dist[1]]
            sloperatio=[mcalc(slope[0],slope[1]),mcalc(slope[1],slope[1]),mcalc(slope[2],slope[1]),mcalc(slope[3],slope[1]),mcalc(slope[4],slope[1])]
            refslope.append(slope[1])
            dist0.append(distratio[0])
            dist2.append(distratio[2])
            dist3.append(distratio[3])
            dist4.append(distratio[4])
            slope0.append(sloperatio[0])
            slope2.append(sloperatio[2])
            slope3.append(sloperatio[3])
            slope4.append(sloperatio[4])
            label.append(dire)
            itr+=1
        if(itr>1000):
            break
    
dic={'slope':refslope,'dist0':dist0,'dist2':dist2,'dist3':dist3,'dist4':dist4,'slope0':slope0,'slope2':slope2,'slope3':slope3,'slope4':slope4,'label':label}
df=pd.DataFrame.from_dict(dic)
df.to_csv('dataset.csv',index=False)

cv2.destroyAllWindows()
# I renamed the folders containing digits to the contained digit itself. For example, digit_0 folder was renamed to 0.
# so taking the 9th value of the folder gave the digit (i.e. "./train/8" ==> 9th value is 8), which was inserted into the first column of the dataset.
        # value = np.hstack((directory[11:],value))
        # df = pd.DataFrame(value).T
        # df = df.sample(frac=1) # shuffle the dataset
        # with open('train_foo.csv', 'a') as dataset:
        #     df.to_csv(dataset, header=False, index=False)