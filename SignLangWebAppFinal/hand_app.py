import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image #pillow
from textblob import TextBlob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from difflib import get_close_matches
import os
from gtts import gTTS


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


DEMO_IMAGE='hand.jpeg'
DEMO_VIDEO='hands.mp4'
mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

df=pd.read_csv('dataset.csv',header=0)
meandf=df.groupby('label').mean()

df1=pd.read_csv('datasetwords.csv',header=0)
meandf1=df1.groupby('label').mean()

st.title('Sign Language App using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

st.sidebar.title('Sign Lang Detection Sidebar')


#We'll create a general func. that resizes the size of image to fit to browswer's window.
#We want to cache this function so that we don't need to call it over and over again:
@st.cache()
def image_resize(image,width=None, height=None, inter=cv2.INTER_AREA):
    dim=None
    (h,w)=image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r=width/float(w)
        dim=(int(w*r),height)

    else:
        r=width/float(w)
        dim=(width,int(h*r))

    #resize the image
    resized=cv2.resize(image,dim,interpolation=inter)

    return resized

app_mode=st.sidebar.radio('Choose the app mode',
                              ['About App','Info about Signs','Run on Image','Run on Video','Info about Project group members']
                              )
if app_mode == 'About App':
    st.markdown("In this project, we're using **mediapipe** for creating a sign language recognition app.")
    #Apply formatting of page width height etc. as above:
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.image('about.png')
    st.markdown("The working of our model in the backend uses the basics of mediapipe framework to recognize the hands of the user. MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3-D landmarks of a hand from just a single frame.")
    st.markdown("These points and lines that we obtain using mediapipe are our key features, upon which we train our ML model. The basic idea is to first choose a reference finger. Then for each sign we compute distance and angles between reference finger and all other fingers. These distances and angles are stored while training for 1000s of images for each sign. Then we take mean of all trained values corresponding to each label and store it as the feature values for that label. Then while testing, our model identifies the angles and points of the hand and processes it. The label is returned which has closest resemblence to the distance and slopes of the sign shown by the user. If model detects incorrectly, we can delete it and repeat the process too.")
    st.markdown("We can form any possible word using the 26 alphabets of English. Additionally, we can generate some common words too using sign language. The dataset for these common words signs is self-created by us. We even have text to speech conversion of the words we formed. We even have speech to text conversion module too for establishing 2 way communication here.")
#................................................................................................
elif app_mode=='Run on Image':
    drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')
    #Apply formatting of page width height etc. as above:
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown("**Predicted output:**")
    kpil_text=st.markdown("")

    max_hands=st.sidebar.number_input("Maximum number of Hands", value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence=st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.7)
    st.sidebar.markdown('---')

    #Now, we need to import our image

    img_file_buffer=st.sidebar.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))
    else:
        demo_image=DEMO_IMAGE #Assigned at the starting of code
        image=np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    hand_count=0
    x , y, c = image.shape
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=max_hands, min_detection_confidence=detection_confidence)
    mpDraw = mp.solutions.drawing_utils
    finlist=[]
    #Dashboard part:
    with mp_hands.Hands(
        static_image_mode=True, #bec. we've unrelated images generally
        max_num_hands=max_hands,
        min_detection_confidence=detection_confidence) as hands:

        result=hands.process(image)
        out_image=image.copy()

        landmarks = []
        for handslms in result.multi_hand_landmarks:
            hand_count+=1
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(out_image, handslms, mpHands.HAND_CONNECTIONS)
        
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
        
        st.subheader('Output image')
        st.image(out_image,use_column_width=True)
        kpil_text.write(f"<h1 style='text-align: center; color:red'>{finlist[0]}</h1>", 
                            unsafe_allow_html=True)
 #................................................................................................
elif app_mode=='Run on Video':
    st.set_option('deprecation.showfileUploaderEncoding', False) #To supress any deprication warning on streamlit
    use_webcam=st.sidebar.button('Use webcam')

    #Apply formatting of page width height etc. as above:
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
    )


    max_hands=st.sidebar.number_input("Maximum number of Hands", value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence=st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.sidebar.markdown('---')

    st.markdown("## Output")
    # Now, we need to learn to work with video data:
    stframe=st.empty()
    video_file_buffer=st.sidebar.file_uploader("Upload a Video",type=["mp4","mov","avi","asf","m4v"])
    tffile=tempfile.NamedTemporaryFile(delete=False)
    
    # Part where we get our input video
    if not video_file_buffer:
        if use_webcam:
            vid=cv2.VideoCapture(0)
        else:
            vid=cv2.VideoCapture(DEMO_VIDEO)
            tffile.name=DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid=cv2.VideoCapture(tffile.name)
    width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input=int(vid.get(cv2.CAP_PROP_FPS))

    #Creating interface for our video:
    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)
    #frame counter:
    fps=0
    i=0 #iteration=0 initialized
    finlist=[]
    
    c=0
    

    loophands=0
    drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    #We'll create 3 columns now:
    kpi1,kpi2,kpi3=st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text=st.markdown("0")

    with kpi2:
        st.markdown("**Detected Hands**")
        kpi2_text=st.markdown("0")

    with kpi3:
        st.markdown("**Frame Width:**")
        kpi3_text=st.markdown("0")

    st.markdown("<hr/>",unsafe_allow_html=True)




    hand_count=0
    framewindow=80
    stri=' '
    #hands Predictor:
    with mp_hands.Hands(
        max_num_hands=max_hands,
        min_detection_confidence=detection_confidence
        ) as hands:
        
        prevTime=0
        while vid.isOpened():
            
            ret,frame=vid.read()
            x , y, c = frame.shape
            if not ret:
                continue

            results=hands.process(frame)
            frame.flags.writeable=True

            hand_count=0
            if results.multi_hand_landmarks:
                loophands=(loophands+1)%framewindow
        
                landmarks = []
                    
                for handslms in results.multi_hand_landmarks:
                    hand_count+=1
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
            
                        landmarks.append([lmx, lmy])
       
                    mp_drawing.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
                i+=1
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
                    if (k=="space"):
                        if (stri[len(stri)-1]!=' ') and (len(stri)>1)  :
                            lastw=lastWord(stri)
                            tranw=translate(lastw)
                            stri=stri[:-len(lastw)]
                            stri=stri+tranw

                    elif k=='del':
                        stri=stri[:-1]
                    elif k=='dot':
                        stri=stri+'.'
                        gfg = TextBlob(stri)
                        stri = str(gfg.correct())
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
                    
            #FPS counter logic
            currTime=time.time()
            fps=1/(currTime-prevTime)
            prevTime=currTime
            
            #Dashboard:
            kpi1_text.write(f"<h1 style='text-align: center; color:red'>{int(fps)}</h1>", 
                                        unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red'>{hand_count}</h1>", 
                                        unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color:red'>{width}</h1>", 
                                        unsafe_allow_html=True)
            
            #Resizing frame to make it fit to window:
            frame=cv2.resize(frame,(0,0), fx=1,fy=1)
            frame=image_resize(image=frame,width=640)
            stframe.image(frame,channels='BGR',use_column_width=True)

        st.text("Video Processed")
        output_video=open("output1.mp4",'rb')
        out_bytes=output_video.read()
        st.video(out_bytes)

        vid.release()
        out.release()

elif app_mode=='Info about Project group members':
    st.header('Project created by:')
    st.subheader('1. Divyansh Bisht')
    st.text('Roll No. 21111027')
    st.subheader('2. Manthan Kojage')
    st.text('Roll No. 21111039')
    st.subheader('3. Manu Shukla')
    st.text('Roll No. 21111040')
    st.subheader('4. Yash Prakash Patil')
    st.text('Roll No. 21111410')

elif app_mode=='Info about Signs':
    st.subheader('Signs and their labels:')
    st.image('1.jpg')
    st.image('2.jpg')
    st.image('3.jpg')
