## Default modules imported. Import more if you need to.

import numpy as np
import random
from random import choice

## Fill out these functions yourself

# Fits a line with Ransac
#   points: Nx2 array of (x,y) pairs of N points
#   K: Number of Samples per run
#   N: Number of runs
#   eps: Error Threshold for outlier
# 
# Return a vector L = (m,b) such that y= mx + b.
#
# Error is same as part a, (y-mx-b)^2
#
# Use ransac to do N runs of:
#   - Sample K points
#   - Fit line to those K points
#   - Find inlier set as those where error is < epsilon
#
# Then pick the solution whose inlier set is the largest,
# and the final solution is the best fit over this inlier set.
#
# You can use np.random.choice to sample indices of points.
def ransac(points, K, N, eps):
    x=points[:,0]
    y=points[:,1]
    i=0
    #sample=random.sample(points,K)
    #y=np.zeros(sample.shape,dtype=np.float32)
    num=0
    globalnum=0
    globalm=0
    globalb=0
    while i<N-1:
        #plist=list(points)
        index=np.random.choice(N,K,replace='False')
        i=i+1

        #sample=points[index,:]


        #sample[index]=np.random.choice(index,K)
        #sample=np.array(lsample)
        tmpx=x[index]
        tmpy=y[index]
        a1 = np.sum(tmpx*tmpx)
        a2 = np.sum(tmpx)
        a4 = K
        a=np.array([[a1,a2],[a2,a4]])
        #print(a)
        c1=np.sum(np.multiply(tmpx,tmpy))
        c2=np.sum(tmpy)
        c=np.array([c1,c2])
        #print(c)
        ans=np.linalg.solve(a,c)
        #print(ans)
        #print("------")

        m=ans[0]
        b=ans[1]
        indexin = np.where(np.abs(y - m * x - b) < np.sqrt(eps))
        # ones=np.ones(x.shape,dtype=np.float32)
        # zeros=np.zeros(x.shape,dtype=np.float32)
        # zeros[indexin]=ones[indexin]
        # num=np.sum(zeros)
        num=indexin[0].shape[0]
        #print(num,num2)
        if(num>globalnum):
            # tmpx=x[indexin]
            # tmpy=y[indexin]
            # a1 = np.sum(tmpx * tmpx)
            # a2 = np.sum(tmpx)
            # a4 = K
            # a = np.array([[a1, a2], [a2, a4]])
            # # print(a)
            # c1 = np.sum(np.multiply(tmpx, tmpy))
            # c2 = np.sum(tmpy)
            # c = np.array([c1, c2])
            # # print(c)
            # ans = np.linalg.solve(a, c)
            # # print(ans)
            # # print("------")
            #
            # m = ans[0]
            # b = ans[1]
            globalm=m
            globalb=b
            globalnum=num
            tmpxp=x[indexin]
            tmpyp=y[indexin]
            a1 = np.sum(tmpxp * tmpxp)
            a2 = np.sum(tmpxp)
            a4 = K
            a = np.array([[a1, a2], [a2, a4]])
            # print(a)
            c1 = np.sum(np.multiply(tmpxp, tmpyp))
            c2 = np.sum(tmpyp)
            c = np.array([c1, c2])
            # print(c)
            ans = np.linalg.solve(a, c)
            # print(ans)
            # print("------")

            m = ans[0]
            b = ans[1]





    return np.float32([globalm,globalb])





########################## Support code below

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Visualize
def vizErr(ax,pts,trueL,estL):
    x = pts[:,0]
    y = pts[:,1]

    ax.hold(True)
    ax.scatter(x,y,s=0.5,c='k')

    x0 = np.float32([np.min(x),np.max(x)])
    y0 = trueL[0]*x0+trueL[1]
    y1 = estL[0]*x0+estL[1]

    ax.plot(x0,y0,c='g')
    ax.plot(x0,y1,c='r')

    g = np.abs(y0[1]-np.sum(y0)/2)
    ax.set_ylim([np.mean(y0)-10*g,np.mean(y0)+10*g])

    return np.sum((y0-y1)**2)/2


rs = np.random.RandomState(0) # Repeatable experiments

# True line and noise free points
trueL = np.float32([1.5,3])
x = rs.uniform(-0.5,0.5,(1000,1))
y = x*trueL[0]+trueL[1]

##### Noisy measurements
# Gaussian Noise
gnz = rs.normal(0,1,(1000,1))
# Outlier Noise
onz = np.float32(rs.uniform(0,1,(1000,1)) < 0.5)

pts = np.concatenate((x,y+(0.025 + 50.0 * onz)*gnz),axis=1)


# Run code and plot errors
eps=0.01 

print("Run this code multiple times!")

ax=plt.subplot(221)
estL = ransac(pts,50,4,eps)
print("(Top Left) K = 50, N = 4, Error = %.2f" % vizErr(ax,pts, trueL, estL))

ax=plt.subplot(222)
estL = ransac(pts,5,40,eps)
print("(Top Right) K = 5, N = 40, Error = %.2f" % vizErr(ax,pts, trueL, estL))

ax=plt.subplot(223)
estL = ransac(pts,50,100,eps)
print("(Bottom Left) K = 50, N = 100, Error = %.2f" % vizErr(ax,pts, trueL, estL))

ax=plt.subplot(224)
estL = ransac(pts,5,1000,eps)
print("(Bottom Right) K = 5, N = 1000, Error = %.2f" % vizErr(ax,pts, trueL, estL))

plt.show()
