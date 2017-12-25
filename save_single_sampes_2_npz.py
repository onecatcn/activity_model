import numpy as np
import math
import os
##import bucket_util as bu
import datetime

class fetch_data():
    def __init__(self):
        self.dir = os.path.expanduser('/home/kai/04_headcount_in_house/01_single_in_house/')
        self.outputdir = os.path.expanduser('/home/kai/04_headcount_in_house/01_single_in_house/')

        # load yield data
        length = 1000
        self.ppnumber = np.linspace(1, 1, length)        
        # generate index for all data
        self.index_all = np.arange(length)
#       print('self.index_all: ',self.index_all)

    def act_addon(self,rec1,rec2):
        rec = rec1 + rec2
        recc = np.where(rec>1, 1, rec)
        return recc

    def save_data(self):
        output_image = np.zeros([self.index_all.shape[0], 5, 60])
        print(output_image.shape)
        output_yield = np.zeros([self.index_all.shape[0]])
        output_index = np.zeros([self.index_all.shape[0]])

        for i in self.index_all:
            if (i%60 == 0):
                print(i,"out of ",len(self.index_all)) 
##            if i % 200 == 0:
##                print ("Saving snapshot!")
##                np.savez(self.outputdir+'histogram_all_full_snapshot_%d.npz' % i,
##                         output_image=output_image,output_yield=output_yield,
##                         output_index=output_index)

            index_temp = str(i)
            s = index_temp.zfill(4)
            filename='singleinhouse'+str(s)+'.npy'

            index = i

            try:
                # This is misleading - it's not just temperature bands - it's the
                # entire multispectral image of depth 9 * times for a single
                # year at a particular location.
                image_temp = np.load(self.dir + filename)
##                print("shape ",image_temp.shape)
                # Filter the image depth to only have the selected days.
                # Days 49-305 are shown here (9 bands each time), but in the paper,
                # the researchers use 49-281.
##                image_temp = self.filter_timespan(image_temp, 49, 305, 10)
##                print (datetime.datetime.now())
##                print (image_temp.shape)

                if np.sum(image_temp) < 3:
                    print ('broken image', filename)

                output_image[i, :] = image_temp
                output_yield[i] = 1
                output_index[i] = i
            except IOError:
                print ("File: %s not found!" % filename)
                continue
            except ValueError:
                print ("File: Value error in %s!" % filename)
                continue
        np.savez(self.outputdir+'histogram_all_full.npz',
                 output_image=output_image,output_yield=output_yield,
                 output_index=output_index)
        print ('save done')

if __name__ == '__main__':
    data=fetch_data()
    data.save_data()
