from django.shortcuts import render,HttpResponse, redirect
from datetime import datetime
# from home.models import Contact
from django.contrib import messages
import json
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators import gzip
import cv2
import mediapipe as mp
import threading
import base64
import pandas as pd
import os
# from handextractor import getletter
import timeit
loophands=0
import math
from difflib import get_close_matches
from gtts import gTTS
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import load_model

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
  


def mcalc(m1,m2):
    pi=math.pi
    pi2=pi*2
    a = m1 - m2
    if a>pi:
        a = pi2-a
    elif a<-pi:
        a += pi2
    return a
my_file = open("samp.txt", "r")
content = my_file.read()
dict = content.split(",")
my_file.close()
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
df=pd.read_csv('dataset.csv',header=0)
meandf=df.groupby('label').mean()
print(meandf)
stri=' '
def getletter(loophands,fram):
    global dict
    # global hands
    global meandf
    global stri
    # print(loophands)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils
    # model = load_model('mp_hand_gesture')
    finlist=[]
    framewindow=10
    c=0
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    # print(BASE_DIR)
    # print(os"b_1_rotate_1.jpeg"))
    frame = cv2.imread("b_1_rotate_1.jpeg")
    x , y, c = frame.shape
    frame = cv2.flip(frame, 1)
    # frame = cv2.flip(frame, 1)
    
    
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    print(result.multi_hand_landmarks)
    
    if  result.multi_hand_landmarks:
        

        loophands=(loophands+1)%framewindow
        
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                
                landmarks.append([lmx, lmy])
            #mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
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
        if loophands!=-1:
            fina=finlist[-framewindow:]
            k=(max(set(fina), key = fina.count))
            if (k=="space"):
                if (stri[len(stri)-1]!=' ') and (len(stri)>1)  :
                    lastw=lastWord(stri)
                    tranw=translate(lastw)
                    stri=stri[:-len(lastw)]
                    stri=stri+tranw

            elif k=='del':
                stri=stri[:-1]
            elif k=='.':
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
            print(k)
        return stri
        # cv2.putText(frame, stri, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
    else:
        print('messed')
        loophands=0
        return stri

# Create your views here.
def index(request):
    #return HttpResponse('This is my homepage self created by me')
    return render(request,'index.html')

@csrf_exempt
def cam(request):
    global loophands
    #return HttpResponse('This is my homepage self created by me')
    if(request.method=='POST'):
        start=timeit.default_timer()
        data=request.POST.get("img")
        data=data.replace('data:image/png;base64,', '')
        data=data.replace(' ','+')
        # decodeit = open('hello_level.jpeg', 'wb')
        # decodeit.write(base64.b64decode(data))
        
        # decodeit.close()
        nparr = np.fromstring(base64.b64decode(data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        r=getletter(loophands,image)
        loophands+=1
        # print(timeit.default_timer()-start)
        return HttpResponse(json.dumps({'data':r}))
    loophands=0
    return render(request,'video.html')

def about(request):
    #return HttpResponse('This is my about page self created by me')
    return render(request,'about.html')

def services(request):
    #return HttpResponse('This is my services page self created by me')
    return render(request,'services.html')

def contact(request):
    #return HttpResponse('This is my contact page self created by me')
    if request.method=="POST":
        name=request.POST.get('name')
        email=request.POST.get('email')
        phone=request.POST.get('phone')
        desc=request.POST.get('desc')
        contact=Contact(name=name,email=email,phone=phone,desc=desc,date=datetime.today())
        contact.save()
        messages.success(request,"Your message has been sent")
    return render(request,'contact.html')


