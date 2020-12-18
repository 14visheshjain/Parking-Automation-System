import pandas as pd
import numpy as np
threshold = 5
from firebase import firebase
firebase = firebase.FirebaseApplication("https://.firebaseio.com/")
def bookslot(plate_number ):
    csv_file = pd.read_excel("plat_to_model.xlsx")
        
    csv_file = csv_file.values
    attributes =[]
   
    #loop through csv list
    number =plate_number
   

    for row in csv_file:
        #if current rows 2nd value is equal to input, print that row
        if number == row[0]:
            attributes = row
            break
   
    if( True ): # check vechicle type
        if(threshold < attributes[2]):
            result = firebase.get("/large slots",None)
            keys = result.keys()
            for key in keys:
                if result[key]=="True":
                    return (key,"False","False")
            return ("False","False","True")
        else:
            result = firebase.get("/small slots",None)
            keys = result.keys()
            for key in keys:
                if result[key]=="True":
                    return (key,"False","False")
            result = firebase.get("/large slots",None)
            keys = result.keys()
            for key in keys:
                if result[key]=="True":
                    return (key,"True","False")
                else:
                    return ("False","False","False")
               
            
