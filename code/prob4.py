## Default modules imported. Import more if you need to.
import numpy as np
from scipy.signal import convolve2d as conv2

# Use these as the x and y derivative filters
fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6
#print(fx)
fy = fx.T


# Compute optical flow using the lucas kanade method
# Use the fx, fy, defined above as the derivative filters
# and compute derivatives on the average of the two frames.
# Also, consider (x',y') values in a WxW window.
# Return two image shape arrays u,v corresponding to the
# horizontal and vertical flow.
def lucaskanade(f1,f2,W):
    fave=(f1+f2)/2
    It=f2-f1
    Ix=conv2(fave,fx,'same','symm')
    Iy=conv2(fave,fy,'same','symm')
    Ixsqu=np.square(Ix)
    Iysqu=np.square(Iy)
    IxIy=np.multiply(Ix,Iy)
    kernal=np.ones((W,W),dtype=np.float32)
    IxIt=np.multiply(Ix,It)
    IyIt = np.multiply(Iy, It)
    sigIx2=conv2(Ixsqu,kernal,'same','symm')
    sigIy2 = conv2(Iysqu, kernal, 'same', 'symm')
    sigIxIy = conv2(IxIy, kernal, 'same', 'symm')
    sigIxIt=conv2(IxIt,kernal,'same','symm')
    sigIyIt = conv2(IyIt, kernal, 'same', 'symm')
    v=(sigIyIt/sigIxIy - sigIxIt/sigIx2)/(sigIxIy/sigIx2-sigIy2/sigIxIy)
    u=(sigIyIt/sigIy2-sigIxIt/sigIxIy)/(sigIx2/sigIxIy-sigIxIy/sigIy2)

    return u,v
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


f1 = np.float32(imread(fn('inputs/frame10.jpg')))/255.
f2 = np.float32(imread(fn('inputs/frame11.jpg')))/255.

u,v = lucaskanade(f1,f2,11)


# Display quiver plot by downsampling
x = np.arange(u.shape[1])
y = np.arange(u.shape[0])
x,y = np.meshgrid(x,y[::-1])
plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')

plt.show()
