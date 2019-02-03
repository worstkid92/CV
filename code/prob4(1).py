## Default modules imported. Import more if you need to.

import numpy as np


## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts):
    p1=pts[:,0:2]
    p2=pts[:,2:4]
    (N,a)=pts.shape
    z=np.ones((N,1))
    p1=np.hstack((p1,z))
    p2=np.hstack((p2,z))
    p=np.zeros((N*3,9))
    p[0:N*3-2:3,3:6]=-p1
    p[0:N*3-2:3,6:9]=p1*np.reshape(p2[:,1],(N,1))
    p[1:N*3-1:3,0:3]=p1
    p[1:N*3-1:3,6:9]=-p1*np.reshape(p2[:,0],(N,1))
    p[2:N*3:3,0:3]=-p1*np.reshape(p2[:,1],(N,1))
    p[2:N*3:3,3:6]=p1*np.reshape(p2[:,0],(N,1))
    u,s,v=np.linalg.svd(p)
    #print(v.shape)
    h=v[8,:]
    H=np.reshape(h,(3,3))
    return H
    

# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def splice(src,dest,dpts):
    #print(H)
    h,w,t=src.shape
    print(w,h)
    spts = np.array([[0, 0], [w-1,0], [0, h-1], [w- 1, h - 1]])
    pts=np.hstack((dpts,spts))
    H = getH(pts)
    print(H)
    print(int(pts[0][1]),int(pts[3][1]))
    for i in range(int(pts[0][0]),int(pts[3][0])):
        for j in range(int(pts[0][1]),int(pts[3][1])):

            dest_c=np.array([i,j,1])
            src_temp=np.dot(H,dest_c)
            #print(src_temp)
            src_temp=src_temp/src_temp[2]
            #print(src_temp[0],src_temp[1])
            if(src_temp[0]<w-1)and(src_temp[0]>=0)and(src_temp[1]<h-1)and(src_temp[1]>=0):
            #if (src_temp[0] < w)and (src_temp[1] < h):

                dest[j][i]=interpolation(src_temp[1],src_temp[0],src)
                print(src_temp[0],src_temp[1],w,h)
                #print(dest[j][i])
            #print(i,j)
    return dest

def interpolation(x,y,img):
    x1=int(x)
    x2=x1+1
    y1=int(y)
    y2=y1+1
    I11=img[x1][y1]
    I12=img[x1][y1+1]
    I21=img[x1+1][y1]
    I22=img[x1+1][y1+1]
    I=I11*(x2-x)*(y2-y)+I21*(x-x1)*(y2-y)+I12*(x2-x)*(y-y1)+I22*(x-x1)*(y-y1)
    #print(I)
    return I-0.0000001

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


simg = np.float32(imread(fn('inputs/p4src.png')))/255.
dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

comb = splice(simg,dimg,dpts)

imsave(fn('outputs/prob4.png'),comb)
