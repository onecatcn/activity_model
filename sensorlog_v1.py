#!/usr/bin/python
import time
import numpy as np
import math
import os
import random

##import bucket_util as bu
##import datetime
length = 1000
dir = os.path.expanduser('/home/kai/04_headcount_in_house/01_single_in_house/')

index_all = np.arange(length)

t0 = 1510272000-60*60
t  = t0

onemin  = 60
onehour = 60*onemin
oneday  = 24*onehour

####################################################################################
## livingroom: 0
## diningroom: 1
## kitchen:    2
## bedroom:    3
## restroom:   4
####################################################################################

for j in index_all :
    
    n=str(j)
    s = n.zfill(4)
    filename = 'singleinhouse'+str(s)+'.npy'
    image_temp = np.load(dir + filename)
    
    for i in range(1,np.size(image_temp,1)):
        ###################################
        ## when time approaches to 7:00 AM, get up
        ###################################
        while(abs(t%oneday-7*onehour)<=5*onemin):
            t += 5*(random.randint(2,5))+((-1)**np.random.randint(0,2))*random.randint(1,60)
        ## put on clothes
        t = active(t,room=3,8min,15min)
        ## morning shower
        t = active(t,room=4,15,30)
        ## breakfast
        t = active(t,room=2&1,25,50)
        ###################################
        ## rest until morning shopping
        ###################################
        while(t%oneday-(9*onehour)<=2*onemin)):
            t = active(t,room=0(80%),all,2-3)
        ## go out for morning shopping
        t = active(t,door)
        ###################################
        ## go back home when time approaches to 11am
        ###################################
        while(abs(t%oneday-11*onehour)<=5*onemin)):
            t += onemin*(random.randint(2,5))+((-1)**np.random.randint(0,2))*random.randint(1,60)
        ## get back and start to prepare for lunch
        t = active(t,door=1)
        ## rest or putting stuff in the fridge
        t = active(t,room=0&1,15,25)
        ## prepare for lunch
        t = active(t,room=1&2,20min,40min)
        ## having lunch
        t = active(t,room=1,20min,40min) 
        ## cleaning after lunch
        t = active(t,room=1&2,10min,30min)
        ## resting at living room before nap
        while(abs(t%oneday-12.5*onehour)<=5*onemin):
            t += onemin*(random.randint(2,5))+((-1)**np.random.randint(0,2))*random.randint(1,60)
            t = active(t,room=0)
        ######################################
        ## napping
        ######################################
        t = active_sleep(t,room=3,60min,120min)
        ## wandering in apt after napping, such as restroom, dining, living, every room
        t0 = t
        while(t-t0<=30*onemin):
            t += onemin*(random.randint(2,5))+((-1)**np.random.randint(0,2))*random.randint(1,60)
            t = active(t,room=(012),)
        ##going out -- afternoon
        t = active(t,door)
        ###################################
        ## go back home when time approaches to 11am
        ###################################
        while(abs(t%oneday-11*onehour)<=16*onemin)):
            t += onemin*(random.randint(2,5))+((-1)**np.random.randint(0,2))*random.randint(1,60)
         ## get back and start to prepare for dinner
        t = active(t,door=1)
        ## rest or putting stuff in the fridge
        t = active(t,room=0&1,15,25)
        ## prepare for lunch
        t = active(t,room=1&2,20min,40min)
        ## having dinner
        t = active(t,room=1,20min,40min) 
        ## cleaning after dinner
        t = active(t,room=1&2,10min,30min)
        ## resting at living room before nap
        t = active(t,room=0,10min,30min)
        ##going out -- afternoon
        t = active(t,door)
        ###################################
        ## go back home when time approaches to 2030am
        ###################################
        while(abs(t%oneday-20.5*onehour)<=16*onemin)):
            t += onemin*(random.randint(2,5))+((-1)**np.random.randint(0,2))*random.randint(1,60)
        ## get back and start to prepare for sleep
        t = active(t,door=1)
        t = active(t,room=0,5,10min)
        ## evening shower
        t = active(t,room=4,15,45)
        ## sleep
        t = active(t,room=3)
