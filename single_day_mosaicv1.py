import numpy as np
import random


####################################################################################
## for one day:
### get up 7:00
## moring shower 30 mins
## breakfast 30 mins
## go out and return at 10:30
### prepare for lunch 40min
## lunch 20min
### clean after lunch 20min
## rest after lunch 30min
## noon nap 1h30m
## afternoon leisure: go out/ tv/ PC
### prepare dinner 4-5
### dinner 5pm-530pm
## clean after dinner 530-600pm
## rest 600-730
## go out 730-830
## evening shower 840-920pm
### sleep 940pm
####################################################################################

##def daily_route():
##    t0 = 1510272000-60*60
##    t  = t0




    
##    t_getup_model = np.datetime64('2017-01-01 07:00:00')
##    t_morning_lunch = np.datetime64('2017-01-01 10:30:00')
##    t_evening_dinner = np.datetime64('2017-01-01 07:00:00')
##    t_morning_sleep = np.datetime64('2017-01-01 10:30:00')
##    
##    t_getup = t_getup_model + ((-1)**np.random.randint(0,2))*(random.randint(0,15)*60+random.randint(0,59))


####################################################################################
## livingroom: 0
## diningroom: 1
## kitchen:    2
## bedroom:    3
## restroom:   4
####################################################################################

####################################################################################
## model one day's PIR records for one person who lives in an apartment  ##
## the designing has several rules ##
## sleep, meals are preprogrammed in the model, though the timing varies ##
## restroom, going out, watch tv are randomly distributed ##
## and most likely to happen ##

## the motion could be random for that the object could rest for a while ##
## the key is the first minute and last minute
## should be 1, for the object has to enter
## and leave the room
####################################################################################


def generate_rand_mild(n, sum_v):
    Vector = [random.randint(1,(2*n)) for i in range(n)]
    Vector = [ int(i / sum(Vector) * sum_v) for i in Vector]
    if sum(Vector) < sum_v:
        Vector[0] += sum_v-sum(Vector)
    Vector[0]=1
    Vector[-1]=1
    return Vector

def generate_rand_extreme(n, sum_v):
    Vector = [random.random() for i in range(n)]
    Vector[Vector.index(min(Vector))]=0
##    print('V',Vector)
    Vector = [ int(i / sum(Vector) * sum_v) for i in Vector]
    if sum(Vector) < sum_v:
        Vector[0] += sum_v-sum(Vector)
    Vector[0]=1
    Vector[-1]=1
    return Vector

def motion_in_1rm(duration):
    Vector = np.random.randint(0,3,(1,duration))
    Vector = np.where(Vector>1,1,0)
    Vector[0,0] = 1
    Vector[0,(duration-1)] = 1
    return Vector


####################################################################################
## living room model ##
####################################################################################
## object would be doing living room activities, such as watching tv or reading
## they will occasionly move out to dining room for water/snacks, or to restroom
## or to bedroom for glasses
####################################################################################
def single_in_house(mainrm,totaltime,totalrm_cnt,max_n_othrm):

####################################################################################
#####  only part of the time will be spend in the main room [ 5t/8 , 7t/8 ]
####################################################################################

    totaltime_in_mainrm = int((totaltime*3/4 + ((-1)**np.random.randint(0,2))\
                               *np.random.randint(totaltime/8)))
    totaltime_in_othrm = totaltime - totaltime_in_mainrm

####################################################################################
#####  the person's path is designed in such a pattern:
#####  M -> O -> M -> O -> M -> O -> M-> O -> M
#####  M/O are random so may be 0 as such the actual pattern may deviate a little from above
#####  time series spent in other room are in extreme mode
#####  time series spent in the main room are in mild mode
####################################################################################

    time_in_othrm = generate_rand_extreme(max_n_othrm,totaltime_in_othrm)
    time_in_mainrm= generate_rand_mild(max_n_othrm+1,totaltime_in_mainrm)
##    print('time_in_othrm',time_in_othrm)
##    print('time_in_mainrm',time_in_mainrm)

####################################################################################
#####  initialize the array, with 1 min as 1 step
####################################################################################
    record = np.zeros((totalrm_cnt,totaltime))
##    record[0,0]=1
    rm_cnt = np.size(time_in_mainrm)
    time_cnt = 0

####################################################################################
#####  fill in the blank, starting from M[0]->O[0]->...M[n]->O[n]->M[n+1]->O[n+1]...
####################################################################################

####################################################################################
######### if rm_cnt=5 then i in [0,1,2,3]
####################################################################################
    for i in range(rm_cnt-1):
        if (time_in_mainrm[i]!=0):

##            print('motion_in_1rm',motion_in_1rm( time_in_mainrm[i] ))
##            print('time part',time_cnt,time_cnt+ time_in_mainrm[i]-1)
##            print('record:',record[mainrm,time_cnt:(time_cnt+ time_in_mainrm[i]-1)]  )
            record[mainrm,time_cnt:(time_cnt+ time_in_mainrm[i])] =\
                                              motion_in_1rm( time_in_mainrm[i] )
        time_cnt+=time_in_mainrm[i]
        

        if (time_in_othrm[i]!=0):
            othrm = i
            if(i>=mainrm):
                othrm = i+1
            record[othrm,time_cnt:(time_cnt+ time_in_othrm[i])] \
                                             = motion_in_1rm( time_in_othrm[i] )
        time_cnt+=time_in_othrm[i]
        
####################################################################################
######## if rm_cnt = 5 then time_in_mainrm[4]
####################################################################################

    if (time_in_mainrm[rm_cnt-1]!=0):   
        record[mainrm,time_cnt:(time_cnt+ time_in_mainrm[rm_cnt-1])] \
                                          = motion_in_1rm( time_in_mainrm[rm_cnt-1] )   


    return record

def act_addon(rec1,rec2):
    rec = rec1 + rec2
    recc = np.where(rec>1, 1, rec)
    return recc


if __name__ == "__main__":

    totaltime   = 60
    totalrm_cnt = 5
    max_t_othrm = 12
    max_n_othrm = 2
    mainrm_num = 0
##    two_in_house = act_addon(single_in_house(np.random.randint(0,totalrm_cnt-2),\
##                                             totaltime,totalrm_cnt),\
##                             single_in_house(np.random.randint(0,totalrm_cnt-2),\
##                                             totaltime,totalrm_cnt))
    output_dir = "/home/kai/04_headcount_in_house/01_single_in_house/"

    for i in range(0,1000):
        n=str(i)
        s = n.zfill(4)
        filename=output_dir+'singleinhouse'+str(s)+'.npy'
        np.save(filename,single_in_house(np.random.randint(0,totalrm_cnt-2),\
                                             totaltime,totalrm_cnt))
        print(filename,':written ' )
    print('two_in_house',two_in_house)


