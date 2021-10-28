from django.shortcuts import render,HttpResponse
from datetime import datetime
from home.models import Contact
from django.contrib import messages

from django.views.decorators import gzip
import cv2
import threading

# Create your views here.
def index(request):
    #return HttpResponse('This is my homepage self created by me')
    return render(request,'index.html')

def cam(request):
    #return HttpResponse('This is my homepage self created by me')
    return render(request,'cam.html')

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
