### EDF --- An Autograd Engine for instruction
## (based on joint discussions with David McAllester)

import numpy as np
from scipy.signal import convolve2d
# Global list of different kinds of components
ops = []
params = []
values = []


# Global forward
def Forward():
    for c in ops: c.forward()

# Global backward    
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: c.backward() 

# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad


## Fill this out        
def init_momentum():
    for p in params:
        p.g=np.zeros_like(p)


    #pass


## Fill this out
def momentum(lr,mom=0.9):
    #pass
    g=init_momentum()
    for p in params:
        p.g=p.grad+mom*p.g
        p.top=p.top-lr*p.g



###################### Different kinds of nodes

# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()


### Operations

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad

# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top, self.y.top)
        #b=np.matmul(self.x.top, self.y.top)
        #print(b)


    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation            
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))



# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS); yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save
        # No backprop to labels!    

# Compute accuracy (for display, not differentiable)        
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        pass


# Downsample by 2    
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)
            
# Convolution Layer
## Fill this out
class conv2:

    def __init__(self,x,k):
        ops.append(self)
        self.x = x
        self.k = k


    def forward(self):
        tmpk=self.k.top
        tmpx=self.x.top

        #print(tmpx.shape)
        #print(tmpk.shape)

        H=tmpk.shape[0]
        W=tmpk.shape[1]
        C1=tmpk.shape[2]
        C2=tmpk.shape[3]
        #print("c2",C2)
        b=tmpx.shape[0]
        y=tmpx.shape[1]
        x=tmpx.shape[2]
        g=np.zeros((b,y-H+1,x-W+1,C2))

        for bi in range(b):
            out=np.zeros((y-H+1,x-W+1,C2))
            for j in range(W):
                for i in range(H):

                    #print("b",b)
                    kernal=tmpk[i,j,:,:]
                    kernal=kernal.reshape((1,C1,C2))
                    #print(kernal.shape)
                    tmpf=tmpx[bi,:,:,:]
                    #print(tmpf.shape)
                    tmp=tmpf[i:y+i+1-H,j:x+j+1-W,:]

                    #print(tmpf.shape)
                    #tmpf=np.roll(tmpf,-i,axis=0)
                    #tmpf=np.roll(tmpf,-j,axis=1)
                    #print(g.shape)
                    tmpsum=np.matmul(tmp,kernal)
                    out+=tmpsum
                    #print("tmpsum",tmpsum.shape)
            g[bi,:,:,:]=out



       #g=g.transpose((2,1,0,3))
        #print(g.shape)
        self.top=g
       # print("-----------------")






    def backward(self):
        #b=self.x.grad.shape[0]
        #C1L=self.x.grad.shape[3]
        #C2L=self.k.top.shape[3]
        if self.x in ops or self.x in params:  # to the input images
            for batch in range(self.x.grad.shape[0]):
                for C1 in range(self.x.grad.shape[3]):
                    sum = np.zeros((self.x.grad.shape[1], self.x.grad.shape[2]))
                    for C2 in range(self.k.top.shape[3]):
                        sum = sum + convolve2d(self.grad[batch, :, :, C2],self.k.top[:, :, C1, C2] * self.k.grad[:, :, C1, C2], 'full')
                    self.x.grad[batch, :, :, C1] = self.x.grad[batch, :, :, C1] + sum

        if self.k in ops or self.k in params:
            for C1 in range(self.k.grad.shape[2]):
                for C2 in range(self.k.grad.shape[3]):
                    for batch in range(self.grad.shape[0]):
                        self.k.grad[:, :, C1, C2] = self.k.grad[:, :, C1, C2] + convolve2d(self.x.top[batch, :, :, C1],np.rot90(self.grad[batch, :, :, C2], 2), 'valid')


class conv2s:####with stride

    def __init__(self,x,k):
        ops.append(self)
        self.x = x
        self.k = k


    def forward(self):
        tmpk=self.k.top
        tmpx=self.x.top

        #print(tmpx.shape)
        #print(tmpk.shape)

        H=tmpk.shape[0]
        W=tmpk.shape[1]
        C1=tmpk.shape[2]
        C2=tmpk.shape[3]
        #print("c2",C2)
        b=tmpx.shape[0]
        y=tmpx.shape[1]
        x=tmpx.shape[2]
        g=np.zeros((b,y-H+1,x-W+1,C2))

        for bi in range(b):
            out=np.zeros((y-H+1,x-W+1,C2))
            for j in range(0,W,2):
                for i in range(0,H,2):

                    #print("b",b)
                    kernal=tmpk[i,j,:,:]
                    kernal=kernal.reshape((1,C1,C2))
                    #print(kernal.shape)
                    tmpf=tmpx[bi,:,:,:]
                    #print(tmpf.shape)
                    tmp=tmpf[i:y+i+1-H,j:x+j+1-W,:]

                    #print(tmpf.shape)
                    #tmpf=np.roll(tmpf,-i,axis=0)
                    #tmpf=np.roll(tmpf,-j,axis=1)
                    #print(g.shape)
                    tmpsum=np.matmul(tmp,kernal)
                    out+=tmpsum
                    #print("tmpsum",tmpsum.shape)
            g[bi,:,:,:]=out



       #g=g.transpose((2,1,0,3))
        #print(g.shape)
        self.top=g
        self.top = self.top[:, ::2, ::2, :]
       # print("-----------------")






    def backward(self):
        #b=self.x.grad.shape[0]
        #C1L=self.x.grad.shape[3]
        #C2L=self.k.top.shape[3]
        if self.x in ops or self.x in params:  # to the input images
            for batch in range(self.x.grad.shape[0]):
                for C1 in range(self.x.grad.shape[3]):
                    sum = np.zeros((self.x.grad.shape[1], self.x.grad.shape[2]))
                    for C2 in range(self.k.top.shape[3]):
                        sum = sum + convolve2d(self.grad[batch, :, :, C2],self.k.top[:, :, C1, C2] * self.k.grad[:, :, C1, C2], 'full')
                    self.x.grad[batch, :, :, C1] = self.x.grad[batch, :, :, C1] + sum

        if self.k in ops or self.k in params:  # to the kernel
            for C1 in range(self.k.grad.shape[2]):
                for C2 in range(self.k.grad.shape[3]):
                    for batch in range(self.grad.shape[0]):
                        self.k.grad[:, :, C1, C2] = self.k.grad[:, :, C1, C2] +convolve2d(self.x.top[batch, :, :, C1],np.rot90(self.grad[batch, :, :, C2], 2), 'valid')


