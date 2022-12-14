import cv2
import os
import time
from gtts import gTTS
from textblob import TextBlob
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from difflib import get_close_matches
my_file = open("samp.txt", "r")
content = my_file.read()
dict = content.split(",")
my_file.close()
def translate(w):
    w=w.lower()
    p=w+" "
    if w in dict:
        return p
    else:
        d=get_close_matches(w, dict,n=1)
        if len(d) ==1:
            asw=d[0]+' '
            return asw
        else:
            return p
def lastWord(strin):
    
    newstring = ""  
    length = len(strin)
    
    
    for i in range(length-1, 0, -1):
        
        if i==1:
            return strin
        
        else:
            if(strin[i] == " "):
            
        
                return newstring[::-1]
            else:
                    newstring = newstring + strin[i]
  

pi=math.pi
pi2=pi*2
def mcalc(m1,m2):
    a = m1 - m2
    if a>pi:
        a = pi2-a
    elif a<-pi:
        a += pi2
    return a

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

df=pd.read_csv('dataset.csv',header=0)
meandf=df.groupby('label').mean()
print(meandf)

df1=pd.read_csv('datasetwords.csv',header=0)
meandf1=df1.groupby('label').mean()
print(meandf1)



# model = load_model('mp_hand_gesture')
i=0
cap = cv2.VideoCapture(0)
finlist=[]
stri=' '
c=0
framewindow=80

loophands=0
for j in range(100000):
    
    
    
    _ , frame = cap.read()
    x , y, c = frame.shape
    # frame = cv2.flip(frame, 1)
    
    
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    
    if  result.multi_hand_landmarks:
        
        
        loophands=(loophands+1)%framewindow
        
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        i+=1 
        print(landmarks)
        if len(landmarks)!=42:
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
            inp=np.array([slope[1],distratio[0],distratio[2],distratio[3],distratio[4],sloperatio[0],sloperatio[2],sloperatio[3],sloperatio[4]])
            final=''
            min=1000
            for ind,row in meandf.iterrows(): 
                dis=np.linalg.norm(inp-np.array(row)) 
                if(dis<min):
                    final=ind
                    min=dis
            finlist.append(final)
        else:
            landmarks1=landmarks[21:]
            landmarks=landmarks[:21]
            landmarks1.pop(1)
            landmarks.pop(1)
            landmarks1=np.array(landmarks1)
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
  

            distt=[]  
            slopee=[]

            for i in range(0,5):
                x=i*4
                li=[np.linalg.norm(landmarks1[x]-landmarks1[x+1]),np.linalg.norm(landmarks1[x]-landmarks1[x+2]),np.linalg.norm(landmarks1[x]-landmarks1[x+3])]
                ind=li.index(max(li))+1
                distt.append(li[ind-1])
                slopee.append(math.atan2(landmarks1[x+ind][1]-landmarks1[x][1],landmarks1[x+ind][0]-landmarks1[x][0]))
            if(distt[1]==0):
                distt=[i+1 for i in distt]
            distratiot=[distt[0]/distt[1],distt[1]/distt[1],distt[2]/distt[1],distt[3]/distt[1],distt[4]/distt[1]]
            sloperatiot=[mcalc(slopee[0],slopee[1]),mcalc(slopee[1],slopee[1]),mcalc(slopee[2],slopee[1]),mcalc(slopee[3],slopee[1]),mcalc(slopee[4],slopee[1])]
            inp=np.array([slope[1],distratio[0],distratio[2],distratio[3],distratio[4],sloperatio[0],sloperatio[2],sloperatio[3],sloperatio[4],slopee[1],distratiot[0],distratiot[2],distratiot[3],distratiot[4],sloperatiot[0],sloperatiot[2],sloperatiot[3],sloperatiot[4]])
            final=''
            min=1000
            for ind,row in meandf1.iterrows(): 
                dis=np.linalg.norm(inp-np.array(row)) 
                if(dis<min):
                    final=' '+ind+' '
                    min=dis
            finlist.append(final)
        
        
        
        if loophands==0:
            fina=finlist[-framewindow:]
            k=(max(set(fina), key = fina.count))
            if (k=="del"):
                if (stri[len(stri)-1]!=' ') and (len(stri)>1)  :
                    lastw=lastWord(stri)
                    tranw=translate(lastw)
                    stri=stri[:-len(lastw)]
                    stri=stri+tranw

            elif k=='space':
                stri=stri[:-1]
            elif k=='dot':
                stri=stri+'.'
                gfg = TextBlob(stri)
                stri = str(gfg.correct())
                stri=stri+" "

            elif k=='enter':
                mytext = stri
                myobj = gTTS(text=mytext, lang='en', slow=False)
                myobj.save("welcome.mp3")
                os.system("welcome.mp3")


            else:
                stri=stri+k
        
        
        cv2.putText(frame, stri, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
    else:
        loophands=0
        cv2.putText(frame, stri, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    for i in range(100000):
        continue

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
