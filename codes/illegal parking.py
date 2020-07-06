import numpy as np
import pandas as pd
import time
import cv2
from firebase import firebase




import urllib 
import json
import requests
def update_entry(plate , slot , type_):
    
    my_data = dict()
    my_data["slot"] = slot
    my_data["plate"] = plate
    my_data["type"] = type_
    json_data = json.dumps(my_data).encode()
    request = urllib.request.Request("https://check-in-bad3b.firebaseio.com/invalid parking.json", data=json_data, method="PATCH")

    try:
        loader = urllib.request.urlopen(request)
    except urllib.error.URLError as e:
        message = json.loads(e.read())
        print(message["error"])
    else:
        print(loader.read())


def correct_text(plate_no):
    plate_no = plate_no.split()
    plate_no = "".join(plate_no)
    ans = str("")
    count = 1
    for ch in plate_no:
        if (ch>='A' and ch<='Z') or (ch>='0'and ch<='9'):
            ans = ans + str(ch)
            count = count+1
    print(ans)
    return ans




def  check_whether_parked_at_another_slot(plate_no):
    from firebase import firebase as firebase2
    firebase_slots = firebase2.FirebaseApplication("https://check-in-bad3b.firebaseio.com/")
        
        
    for slot_type in np.array(["large slots","small slots"]):
        slots_in_particular = firebase_slots.get("/"+str(slot_type),None)
        
        for slot in slots_in_particular.keys() :
            if(slots_in_particular[slot]=="False"):
                continue
            import xyz
            print("C:\\Users\\VISHESH JAIN\\Desktop\\sih\\final model\\footage\\"+str(slot_type)+"\\"+str(slot)+".mp4")
#            iscar , detected_plate , isplate = xyz.get_text_of_plate("C:\\Users\\VISHESH JAIN\\Desktop\\sih\\final model\\footage"+str(slot_type)+"\\"+str(slot)+".mp4")
            detected_plate = xyz.get_text_of_plate("C:\\Users\\VISHESH JAIN\\Desktop\\sih\\final model\\footage\\"+str(slot_type)+"\\"+str(slot)+".mp4")
            isplate = True
            detected_plate = correct_text(detected_plate)
            if(isplate and plate_no == detected_plate):    
                return ("parked at slot",str(slot))
    return ("not parked at another slot","None")



while True:
    firebase1 = firebase.FirebaseApplication("https://hackitout-6ee0b.firebaseio.com/")

    t= time.localtime()
    curr_time =int(t[3]*60+t[4]) 
    result = firebase1.get("/time",None)
    keys = list(result.keys())
   
    print(keys)
    if len(keys)>0 :#and int(keys[0] )+ int(15) <=int( curr_time) and int(int(keys[0] +int(20))>=int(curr_time)) :
    

        status_of_parking = ""
        
        import xyz
        import is_vehicle


        if(is_vehicle.is_vehicle("C:\\Users\\VISHESH JAIN\\Desktop\\sih\\final model\\4.mp4")=="True" or True ):


            slot_plate_no = xyz.get_text_of_plate("C:\\Users\\VISHESH JAIN\\Desktop\\sih\\final model\\4.mp4")

            ####################changing slot_plate_no_ just for testing purpose
            #slot_plate_no=""
            
            get_key  =  firebase1.get("/time/"+str(keys[0]),None) 
            get_plate  = firebase1.get("/Car Parking/"+str(t[2])+"-"+str(t[1])+"-"+str(t[0])+"/"+str(get_key) +"/carNumber", None)
            
            slot_plate_no=correct_text(slot_plate_no)
            
            print(slot_plate_no,get_plate)
            if(slot_plate_no ==get_plate and False):
                print("parked at correct slot")
                continue

            firebase3 = firebase.FirebaseApplication("https://check-in-bad3b.firebaseio.com/")
            slot  = firebase1.get("/Car Parking/"+str(t[2])+"-"+str(t[1])+"-"+str(t[0])+"/"+str(get_key) +"/slot", None)

            get_neighbours  =  list(firebase3.get("/neighbours/"+slot,None).keys())
            print(get_neighbours)
            for neighbouring_slots in get_neighbours:

                make_path = ""
                if neighbouring_slots[0]=='l':
                    make_path = make_path+str("large slots")
                else:
                    make_path = make_path+str("small slots")
                print("C:\\Users\\VISHESH JAIN\\Desktop\\sih\\final model\\footage"+make_path+"\\"+neighbouring_slots+".mp4")
                is_neighbour_slot_filled  =   firebase3.get(make_path+"/"+neighbouring_slots,None)
                print(is_neighbour_slot_filled )
                if( is_neighbour_slot_filled =="True" and is_vehicle.is_vehicle("C:\\Users\\VISHESH JAIN\\Desktop\\sih\\final model\\footage\\"+make_path+"\\"+neighbouring_slots+".mp4")=="True"):
                    update_entry(get_plate ,neighbouring_slots , "invalidly occupied other slot also")
                    continue


           
                               

            
        else:
            get_key  =  firebase1.get("/time/"+str(keys[0]),None) 
            get_plate  = firebase1.get("/Car Parking/"+str(t[2])+"-"+str(t[1])+"-"+str(t[0])+"/"+str(get_key) +"/carNumber", None)
            status_of_parking = check_whether_parked_at_another_slot(get_plate)



            print(status_of_parking)
            if(status_of_parking[0] == "parked at slot"):
                update_entry(get_plate , status_of_parking[1] , status_of_parking[0])


        #firebase1.delete("/time",keys[0])
            
        

    

        
        

