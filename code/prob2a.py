## Default modules imported. Import more if you need to.

import numpy as np
import matplotlib

## Fill out these functions yourself

# Fits a line iteratively
#   points: Nx2 array of (x,y) pairs of N points
#   eps: Error Threshold
#   numit: Number of iterations
#
# Return a vector L = (m,b) such that y= mx + b.
#
# Fit to minimize  the sum of \|y - mx - b\|^2 over an inlier set, where the
# inlier set is defined as points where the above square error is less than eps.
#
# In iteration 1, all points should be inliers, so that if this function is called
# with numit=1, you should return the best fit over ALL points.
#
# Should you ever hit an inlier set with fewer than 2 elements, simply return the current estimate of L.
def fitLine(points, eps, numit=10):
    y=points[:,1]
    x=points[:,0]
    L=x.shape[0]

    a1=np.sum(x*x)
    a2=np.sum(x)
    a4=L
    #ones=np.ones(imshape,dtype=np.float32)
    a=np.array([[a1,a2],[a2,a4]])
    c1=np.sum(y*x)
    c2=np.sum(y)
    c=np.array([c1,c2])
    ans=np.linalg.solve(a,c)
    m=ans[0]
    b=ans[1]
    i=0
    #np.linalg.norm(y-m*points-b)>np.sqrt(eps)
    while i<numit-1:
        index=np.where(np.abs(y-m*x-b)<np.sqrt(eps))
        #tmpx=np.zeros(x.shape,dtype=np.float32)
        #tmpy=np.zeros(y.shape,dtype=np.float32)
        tmpx=x[index]
        tmpy=y[index]
        a1=np.sum(np.square(tmpx))
        a2=np.sum(tmpx)
        a4=index[0].shape[0]
        c1=np.sum(np.multiply(tmpx,tmpy))
        c2=np.sum(tmpy)
        a = np.array([[a1, a2], [a2, a4]])
        c=np.array([[c1],[c2]])
        ans=np.linalg.solve(a,c)
        m=ans[0]
        b=ans[1]
        i=i+1




    return np.float32([m,b])


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
onz1 = np.float32(rs.uniform(0,1,(1000,1)) < 0.1)
onz2 = np.float32(rs.uniform(0,1,(1000,1)) < 0.5)

# Only Gaussian Noise
pts1 = np.concatenate((x,y+0.025*gnz),axis=1)

# Different percentage of outliers
pts2 = np.concatenate((x,y+(0.025 + 50.0 * onz1)*gnz),axis=1)
pts3 = np.concatenate((x,y+(0.025 + 50.0 * onz2)*gnz),axis=1)


# Run code and plot errors
eps=0.01

ax=plt.subplot(221)
estL = fitLine(pts1,eps,1)
print("(Top Left) No outliers, simple fit Error = %.2f" % vizErr(ax,pts1, trueL, estL))

ax=plt.subplot(222)
estL = fitLine(pts2,eps,1)
print("(Top Right) 10pc outliers, simple fit Error = %.2f" % vizErr(ax,pts2, trueL, estL))

ax=plt.subplot(223)
estL = fitLine(pts2,eps,10)
print("(Bottom Left) 10pc outliers, 10 iters fit Error = %.2f" % vizErr(ax,pts2, trueL, estL))

ax=plt.subplot(224)
estL = fitLine(pts3,eps,10)
print("(Bottom Right) 50pc outliers, 10 iters fit Error = %.2f" % vizErr(ax,pts3, trueL, estL))


plt.show()
