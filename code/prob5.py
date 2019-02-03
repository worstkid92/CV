## Default modules imported. Import more if you need to.

import numpy as np


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################





## Fill out these functions yourself

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img):

    #W = img.shape[1]
    #H = img.shape[0]
    #list=[]
    num=np.zeros(img.shape,dtype=np.uint32)
    for i in range (-2,3):
        for j in range (-2,3):
            if(i==0 and j==0):
                continue
            tmp=img
            #print(tmp[100][100],tmp[99][100],tmp[101][100])
            tmp=np.roll(tmp,i,axis=0)
            tmp=np.roll(tmp,j,axis=1)
            #print(tmp[100][100], tmp[99][100], tmp[101][100])
            #print(i,j)
            tmp[i:0,j:0]=0
            index=np.where(tmp<img)
            #np.left_shift(num,1)
            num=num*2
            num[index]=num[index]+1

    #print(num[100][200])
    #print(num)

    return num
    

# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).
def smatch(left,right,dmax):
    W = left.shape[1]
    H = left.shape[0]
    cenleft=census(left)
    #print(cenleft[100][200])
    #cenright=census(right)
   # distleft=hamdist(cenleft,cenright)
    d=np.zeros((H,W,dmax+1),dtype=np.uint32)
    index=0
    for i in range( dmax+1 ):
        tmp=right
        print(index)
        shifttmp=np.roll(tmp,i,axis=1)
        #compright=shifttmp[0:W-i,:]
        #tmpleft=left[i:,:]
        cenright=census(shifttmp)
        tmpdis=hamdist(cenleft,cenright)
        d[:,:,index]=tmpdis
        index+=1




    ansd = np.argmin(d,axis=2)

   # print(ansd)

    return ansd
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

left = imread(fn('inputs/left.jpg'))
# print(left[100][100])
# left[100][100]=0
# ans=census(left)
# print(ans[100][100])

right = imread(fn('inputs/right.jpg'))

d = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
