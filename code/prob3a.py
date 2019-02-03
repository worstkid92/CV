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


# Copy this from solution to problem 2.
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

def buildcv(left,right,dmax):
    index=0
    cenleft=census(left)
    cv = 24 * np.ones([left.shape[0], left.shape[1], dmax + 1], dtype=np.float32)
    for d in range(dmax+1):
        tmpright=right
        shiftright=np.roll(tmpright,d,axis=1)
        #shiftright[:,0:d]=24
        cenright=census(shiftright)
        tmpd=hamdist(cenleft,cenright)
        cv[:,:,index]=tmpd
        cv[:,0:d,index]=24
        index=index+1

    return cv

# Implement the forward-backward viterbi method to smooth
# only along horizontal lines. Assume smoothness cost of
# 0 if disparities equal, P1 if disparity difference <= 1, P2 otherwise.
#
# Function takes in cost volume cv, and values of P1 and P2
# Return the disparity map
def viterbilr(cv,P1,P2):
    lamda=1
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]
    #S=np.zeros((D,D),dtype=np.float32)
    d1 = np.arange(D).reshape([D, 1])
    d2 = np.arange(D).reshape([1, D])
    S = np.abs(d1 - d2).astype(np.float32)
    index_p1 = S == 1
    index_p2 = S > 1
    S[index_p1] = P1
    S[index_p2] = P2


    d=np.zeros((H,W),dtype=np.int32)
    chat=np.zeros(cv.shape,dtype=np.float32)#cv hat
    chat[:,0,:]=cv[:,0,:]
    z=np.zeros(cv.shape,dtype=np.float32)
    for x in range (W-1):
        for dp in range (D):
            ls = np.zeros((H,D), dtype=np.float32)
            ls[:,np.arange(D)]=chat[:,x,np.arange(D)]+S[np.arange(D),dp]
            #for i in range(D):
             #   if(dp==i):
              #      ls[:,i]=0+chat[:,x,i]
               # elif (np.absolute(dp-i)==1):
                #    ls[:,i]=lamda*P1+chat[:,x,i]
                #else:
                 #   ls[:,i]=lamda*P2+chat[:,x,i]
            #z[:,x+1,dp]
            ttt=np.argmin(ls,axis=1)
            z[:, x + 1, dp]=ttt
            #print(ttt)

            #chat[:,x+1,dp]=cv[:,x+1,dp]+np.amin(ls,axis=1)
            chat[:, x + 1, dp] = cv[:, x + 1, dp] + ls[np.arange(H),ttt]
    d[:,W-1]=np.argmin(chat[:,W-1,:],axis=1)
    print(d[:,W-1])
    for i in range (W-1):
        #print(d[:,W-1-i])
        d[:,W-2-i]=z[np.arange(H),W-1-i,d[:,W-1-i]]
        #for j in range(H):
        #    d[j,W-2-i]=z[j,W-1-i,d[j,W-1-i]]
    #print(d)

    return d
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv = buildcv(left_g,right_g,50)
d = viterbilr(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3a.jpg'),dimg)
