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
# Copy this from problem 2 solution.
def buildcv(left,right,dmax):
    index = 0
    cenleft = census(left)
    cv = 24 * np.ones([left.shape[0], left.shape[1], dmax + 1], dtype=np.float32)
    for d in range(dmax + 1):
        tmpright = right
        shiftright = np.roll(tmpright, d, axis=1)
        # shiftright[:,0:d]=24
        cenright = census(shiftright)
        tmpd = hamdist(cenleft, cenright)
        cv[:, :, index] = tmpd
        cv[:, 0:d, index] = 24
        index = index + 1

    return cv


# Do SGM. First compute the augmented / smoothed cost volumes along 4
# directions (LR, RL, UD, DU), and then compute the disparity map as
# the argmin of the sum of these cost volumes. 
def SGM(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]
    d1 = np.arange(D).reshape([D, 1])
    d2 = np.arange(D).reshape([1, D])
    S = np.abs(d1 - d2).astype(np.float32)
    index_p1 = S == 1
    index_p2 = S > 1
    S[index_p1] = P1
    S[index_p2] = P2


    clr=cv
    crl = cv
    cdu =cv
    cud = cv
    d=np.zeros((H,W),dtype=np.float32)
    for x in range (1,W):
        for dp in range (D):
            ls = np.zeros((H,D), dtype=np.float32)
            ls[:, np.arange(D)] = clr[:, x-1, np.arange(D)] + S[np.arange(D), dp]
            #for i in range(D):
             #   if(dp==i):
              #      ls[:,i]=0+clr[:,x-1,i]
               # elif (np.absolute(dp-i)==1):
                #    ls[:,i]=P1+clr[:,x-1,i]
                #else:
                 #   ls[:,i]=P2+clr[:,x-1,i]
            #z[:,x+1,dp]
            #ttt=np.argmin(ls,axis=1)
            #z[:, x + 1, dp]=ttt
            #print(ttt)

            clr[:,x,dp]=cv[:,x,dp]+np.amin(ls,axis=1)

    for x in reversed(range(W-2,-1)):
        for dp in range (D):
            ls = np.zeros((H,D), dtype=np.float32)
            ls[:, np.arange(D)] = crl[:, x+1, np.arange(D)] + S[np.arange(D), dp]
            #for i in range(D):
             #   if(dp==i):
              #      ls[:,i]=0+crl[:,x+1,i]
               # elif (np.absolute(dp-i)==1):
                #    ls[:,i]=P1+crl[:,x+1,i]
                #else:
                 #   ls[:,i]=P2+crl[:,x+1,i]
            #z[:,x+1,dp]
            #ttt=np.argmin(ls,axis=1)
            #z[:, x + 1, dp]=ttt
            #print(ttt)

            crl[:,x,dp]=cv[:,x,dp]+np.amin(ls,axis=1)
    for y in range(1, H):
            for dp in range(D):
                    ls = np.zeros((W, D), dtype=np.float32)
                    #ls[:, np.arange(D)] = cdu[y-1, :, np.arange(D)] + S[np.arange(D), dp]
                    for i in range(D):
                        if (dp == i):
                            ls[:, i] = 0 + cdu[y-1, :, i]
                        elif (np.absolute(dp - i) == 1):
                            ls[:, i] = P1 + cdu[y-1, :, i]
                        else:
                            ls[:, i] = P2 + cdu[y-1,:, i]
                    # z[:,x+1,dp]
                    # ttt=np.argmin(ls,axis=1)
                    # z[:, x + 1, dp]=ttt
                    # print(ttt)

                    cdu[y, :, dp] = cv[y,: , dp] + np.amin(ls, axis=1)

    for y in reversed(range(H - 2, -1)):
            for dp in range(D):
                    ls = np.zeros((W, D), dtype=np.float32)
                    ls[:, np.arange(D)] = cud[y+1, :, np.arange(D)] +  S[np.arange(D), dp]
                  #  for i in range(D):
                   #     if (dp == i):
                    #        ls[:, i] = 0 + cud[y+1,:, i]
                     #   elif (np.absolute(dp - i) == 1):
                      #      ls[:, i] = P1 + cud[y+1,:, i]
                       # else:
                        #    ls[:, i] = P2 + cud[y+1,:, i]
                    # z[:,x+1,dp]
                    # ttt=np.argmin(ls,axis=1)
                    # z[:, x + 1, dp]=ttt
                    # print(ttt)

                    cud[y,:, dp] = cv[y, :, dp] + np.amin(ls, axis=1)
    tmpcv=clr+crl+cud+cdu
    d=np.argmin(tmpcv,axis=2)

            #chat[:, x + 1, dp] = cv[:, x + 1, dp] + ls[np.arange(H),ttt]

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
d = SGM(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3b.jpg'),dimg)
