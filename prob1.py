## Default modules imported. Import more if you need to.
### Problem designed by Abby Stylianou

import numpy as np
from scipy.signal import convolve2d as conv2


def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.
    cluster_centers = np.zeros((num_clusters,2),dtype='int')
    num=int(np.sqrt(num_clusters))
    l=int(im.shape[0]/num)
    index=0
    grad=get_gradients(im)
    for i in range(num):
        for j in range(num):
            cluster_centers[index,0]=i*l+int(l/2)
            cluster_centers[index, 1] = j * l + int(l / 2)
            index+=1


    for i in range(num_clusters):
        tmpx=cluster_centers[i][0]
        tmpy=cluster_centers[i][1]
        s=grad[tmpx-1:tmpx+2,tmpy-1:tmpy+2]
        index=np.unravel_index(np.argmin(s, axis=None), s.shape)
        index=index-np.array([1,1])
        cluster_centers[i][0]+=index[0]
        cluster_centers[i][1] += index[1]


    return cluster_centers






def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h,w,c = im.shape
    weight=0.8#weight can be changed here
    clusters = np.zeros((h,w))
    mindismap=np.ones((h,w))*1000
    minindex=np.zeros((h,w))
    num=int(np.sqrt(num_clusters))
    l=int(im.shape[0]/num)
    imgwindex=np.zeros((h,w,5))
    imgwindex[:,:,0:3]=im
    a=np.meshgrid(np.arange(h),np.arange(h))
    imgwindex[:,:,3]=a[0]*weight
    imgwindex[:,:,4]=a[1]*weight
    tmplst=[]
    k=0


    for i in range(num):
       for j in range(num):
          tmp=im[i*l:(i+1)*l,j*l:(j+1)*l,:]
          Rmean=np.sum(tmp[:,:,0])/(l*l)
          Gmean=np.sum(tmp[:,:,1])/(l*l)
          Bmean=np.sum(tmp[:,:,2])/(l*l)
          tmprgb=np.array([Rmean,Gmean,Bmean,i*l+int(l/2),j * l + int(l / 2)])
          tmplst.append(tmprgb)


    for i in range(num_clusters):
        miu=tmplst[i]

        tmpx=cluster_centers[i][0]
        tmpy=cluster_centers[i][1]
        miu=miu.reshape((1,1,5))##1*1*5,R,G,B,X,Y
        #print(miu.shape)
        grid=imgwindex[np.maximum(0,tmpx-l):np.minimum(w,tmpx+l),np.maximum(0,tmpy-l):np.minimum(w,tmpy+l),:]##2S*2S area,within boundary
        #print(grid.shape)
        copy=mindismap.copy()
        tmpdis=np.linalg.norm(grid-miu,axis=2)
        copy[np.maximum(0,tmpx-l):np.minimum(w,tmpx+l),np.maximum(0,tmpy-l):np.minimum(w,tmpy+l)]=tmpdis
        index=np.where(copy<mindismap)
        mindismap[index]=copy[index]
        minindex[index]=i


    while(True):
        k=k+1

        labelpixels=[]
        labelpixels.clear()
        indexlst=[]


        for i in range(num_clusters):
            index=np.where(minindex==i)
            indexlst.append(index)
            numpixel=index[0].shape[0]
            if (numpixel == 0):
                labelpixels.append(tmplst[i])
                continue

            distric=im[index[0],index[1],:]
            Rmean=np.sum(distric[:,0])/numpixel
            Gmean = np.sum(distric[:, 1]) / numpixel
            Bmean = np.sum(distric[:, 2]) / numpixel
            xmean=np.sum(index[0])/numpixel
            ymean=np.sum(index[1])/numpixel


            labelpixels.append(np.array([Rmean,Gmean,Bmean,xmean,ymean]))

        sum=0

        for i in range(num_clusters):
            sum+=np.linalg.norm(labelpixels[i]-tmplst[i])

        if(sum<1):
            break

        for i in range(num_clusters):
            miu=labelpixels[i]
            tmpx=int(miu[3])
            tmpy=int(miu[4])
            miu = miu.reshape((1, 1, 5))
            tmpindex=indexlst[i]
            #grid=imgwindex[tmpindex[0],tmpindex[1],:]
            grid = imgwindex[np.maximum(0, tmpx - l):np.minimum(w, tmpx + l),np.maximum(0, tmpy - l):np.minimum(w, tmpy + l), :]
            copy = mindismap.copy()
            tmpdis = np.linalg.norm(grid - miu, axis=2)

            copy[np.maximum(0, tmpx - l):np.minimum(w, tmpx + l),np.maximum(0, tmpy - l):np.minimum(w, tmpy + l)]=tmpdis
            update = np.where(copy < mindismap)

            mindismap[update] = copy[update]
            minindex[update] = i

        tmplst=labelpixels










    return minindex



















########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/lion.jpg')))

num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/prob1b_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
