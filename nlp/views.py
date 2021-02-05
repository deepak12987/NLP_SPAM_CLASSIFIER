from django.shortcuts import render
import pickle
import numpy as np
def home(request):
    return render(request,'main.html')
def pre(request):
    answer = ''
    prob = 0.0
    djtext = request.POST.get("text","default")
    copy =  djtext
    transformer = pickle.load(open("transformer.pkl",'rb'))
    model = pickle.load(open("nlp_spam.pkl","rb"))
    sent = transformer.transform([djtext])
    y = model.predict(sent)
    k = model.predict_proba(sent)
    if y == 0: 
        answer = 'NOT A SPAM MESSAGE'
        prob = k[0][0]
    else : 
        answer = 'SPAM ALERT !!'
        prob = k[0][1]
    params={'predicted':answer,'original':copy,'percentage':int(prob*100)}
    return render(request,'output.html',params)

def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')
