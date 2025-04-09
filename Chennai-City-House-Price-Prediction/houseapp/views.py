
from django.shortcuts import render
from django.shortcuts import redirect
# Create your views here.

def home(req):
    return render(req,"home.html")




# load model....
import pickle 
import numpy as np
import pandas as pd
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
      data2 = pickle.load(file)
    return data2
data2 = load_model()
model2 = data2["model"]
le_status = data2['le_status']
le_location = data2['le_location']
le_builder = data2['le_builder']

def predict(req):
    
    data = pd.read_csv("clean_data.csv")
    data_count = data["location"].value_counts()
    location_count_less_10 = data_count[data_count <= 20]

    data["location"] = data['location'].apply(lambda x:'other' if x in location_count_less_10 else x)
    locations = data["location"].unique()

    # print(locations)
    # print(type(locations))

    builders  = data["builder"].unique()
    data = {"location" : locations,
            "builder" : builders}
    
    if req.method=="POST":
        area = req.POST.get("area",False)
        status = req.POST.get("status",False)
        bhk = req.POST.get("bhk",False)
        bathroom = req.POST.get("bathroom",False)
        age = req.POST.get("age",False)
        
        location = req.POST.get("location",False)
        builder = req.POST.get("builder",False)


        X= np.array([[area,status,bhk,age,bathroom,location,builder]])
        X[:,1] = le_status.transform(X[:,1])
        X[:,5] = le_location.transform(X[:,5])
        X[:,6] = le_builder.transform(X[:,6])
        x= X.astype(float)
        price = model2.predict(X)
        
        price = round(price[0],2)
        print(type(price))
        context = {"price":price}   
        return render(req,"predictprice.html",context)
    return render(req,"predict.html",data)   


def about(req):
    return render(req,"about.html")