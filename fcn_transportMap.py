# This is Matt's original code since transform_tfMatt.py is being used for experiments

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import genData

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

tf.enable_eager_execution()

class GeneralDimension:
    def __init__(self,d,layerSizes, monoLayer):
        numLayers = len(layerSizes)
        self.d = d

        if self.d==0:
            self.bs = monoLayer.add_variable('g_bs_%d_0'%d, shape=[1])

        else:
            self.ws = [None]*(numLayers)
            self.bs = [None]*(numLayers)

            self.ws[0] = monoLayer.add_variable('g_ws_%d_0'%d, shape=[d+1,layerSizes[0]])
            self.bs[0] = monoLayer.add_variable('g_bs_%d_0'%d, shape=[layerSizes[0]])

            for i in range(1,numLayers):
                self.ws[i] = monoLayer.add_variable('g_ws_%d_%d'%(d,i),shape=[layerSizes[i-1],layerSizes[i]])
                self.bs[i] = monoLayer.add_variable('g_bs_%d_%i'%(d,i), shape=[layerSizes[i]])

            self.wlast = monoLayer.add_variable('wlast_%d'%d, shape=[layerSizes[-1],1])
            self.blast = monoLayer.add_variable('blast_%d'%d, shape=[1])

    def Evaluate(self, x):
        if(self.d==0):
            return self.bs

        numLayers = len(self.ws)
        xLayers = [None]*numLayers

        #print ('x shape: {}, ws[0] shape: {}, d: {}'.format(x.shape, self.ws[0].shape, self.d))
        xLayers[0] = tf.nn.relu(tf.matmul(x, self.ws[0]) + self.bs[0])
        for i in range(1,numLayers):
            xLayers[i] = tf.nn.relu(tf.matmul(xLayers[i-1], self.ws[i]) + self.bs[i])

        return tf.matmul(xLayers[-1], self.wlast) + self.blast

class MonotoneDimension:
    def __init__(self,d,layerSizes,monoLayer):
        numLayers = len(layerSizes)
        self.d = d
        #this is the overall topology of the network
        self.ws = [None]*(numLayers+1)
        self.bs = [None]*(numLayers+1)
        #print ('ws: ', self.ws, ' bs: ', self.bs)
        # set up a left endpoint rule for the integration
        self.numQuad = 20
        self.quadPts = np.linspace(0,1,self.numQuad+1)[0:self.numQuad]
        self.quadWts = (self.quadPts[1]-self.quadPts[0])*np.ones(self.numQuad)
        #print ('quad pts: ', self.quadPts, ' self.quadWts: ', self.quadWts)
        # Set up the weightsc
        self.wd = monoLayer.add_variable("wd_%d"%d, shape = [ 1,layerSizes[0]])
        #print ("wd_%d"%d)
        if(d!=0):
            #print ('size_0: [', d, ', ', layerSizes[0], ']')
            self.ws[0] = monoLayer.add_variable('ws_%d_0'%d, shape=[ d,layerSizes[0]])
            self.bs[0] = monoLayer.add_variable('bs_%d_0'%d, shape=[ layerSizes[0] ])

        for i in range(1,numLayers):
            #print ('size_{}: [{}, {}]'.format(i, layerSizes[i-1], layerSizes[i]))
            self.ws[i] = monoLayer.add_variable('ws_%d_%d'%(d,i), shape=[ layerSizes[i-1], layerSizes[i] ])
            self.bs[i] = monoLayer.add_variable('ws_%d_%d'%(d,i), shape=[layerSizes[i]])

        # finally, combine everything into a scalar
        #print ('size_{}: [{}, {}]'.format(i, layerSizes[i-1], 1))
        self.wlast = monoLayer.add_variable('wlast_%d'%d, shape=[layerSizes[-1],1])
        self.blast = monoLayer.add_variable('blast_%d'%d, shape=[1])

    def Evaluate(self,x):
        numSamps = x.shape[0]
        numQuad = self.quadPts.shape[0]
        numLayers = len(self.ws)-1

        xHead = None
        if(self.d!=0):
            xHead = tf.slice(x,[0,0], [numSamps,self.d])
        xTail = tf.slice(x,[0,self.d], [numSamps,1])

        temp=0.0
        if(self.d!=0):
            temp = tf.matmul(xHead,self.ws[0]) + self.bs[0]

        quadVal = 0.0
        for qInd in range(numQuad):
            xLayers = [None]*numLayers

            xLayers[0] = tf.nn.relu(tf.matmul(self.quadPts[qInd]*xTail, self.wd) + temp)

            for i in range(1,numLayers):
                xLayers[i] = tf.nn.relu(tf.matmul(xLayers[i-1], self.ws[i]) + self.bs[i])

            f = tf.matmul(xLayers[-1], self.wlast) + self.blast
            temp2 = tf.square(f)
            quadVal += self.quadWts[qInd]*temp2

        # This multiplication is for change of variable
        quadVal = tf.multiply(quadVal, xTail)

        return quadVal


class MonotoneLayer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(MonotoneLayer, self).__init__()
        self.num_outputs = dim
        self.dim = dim

        layerSizes = [4,4]
        self.monoParts = [ MonotoneDimension(d, layerSizes, self) for d in range(dim)]
        self.genParts = [ GeneralDimension(d, layerSizes, self) for d in range(dim)]
        print ('monoParts: ', self.monoParts)
    def build(self, input_shape):
        print('input_shape = ', input_shape)
        assert input_shape[-1] == self.num_outputs
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        print ('IS THIS HAPPENING??')
        return tf.concat([self.monoParts[d].Evaluate(input) + self.genParts[d].Evaluate(input) for d in range(self.dim)], 1)

numSamps = 8000
halfNumSamps = int(0.5*numSamps)
x1a = np.random.randn(halfNumSamps,1).astype('f')
x1b = np.random.randn(halfNumSamps,1).astype('f') + 2

x2a = np.cos(x1a) + 0.3*np.random.randn(halfNumSamps,1).astype('f')
x2b = np.cos(x1b) - 2.0 + 0.2*np.random.randn(halfNumSamps,1).astype('f')

x1 = x1a;#np.concatenate([x1a,x1b])
x2 = x2a;#np.concatenate([x2a,x2b])
#xnump = np.hstack([x1.reshape(-1,1),x2.reshape(-1,1)])

#x = tf.constant(xnump)

# plt.scatter(xnump[:,0],xnump[:,1],alpha=0.1)
# plt.show()
# quit()
#print ('xnump: ', xnump.shape)
xnump = genData.Banana(halfNumSamps)
print ('xnump: ', xnump.shape)
x = tf.constant(xnump)
monoLayer = MonotoneLayer(xnump.shape[1])
dim = xnump.shape[1]
#x = np.random.randn(numSamps,1)
#y = x*x*x + x

#model.fit(x,y,epochs=10,batch_size=100)

#
#dataset = tf.data.Dataset.from_tensor_slices((x,y))
# print(x[0,:])
# print(y[0,:])

class GaussianKL:
    def __init__(self,d):
        self.d = d

    def __call__(self):

        numSamps = x.get_shape().as_list()[0]
        #xslice = tf.slice(x,[0,d],[numSamps,1])

        with tf.GradientTape() as g:
            g.watch(x)
            #we only use d-1 columns of x for each h() and g() functions (also used in evaluation below)
            r = monoLayer.genParts[self.d].Evaluate(x) + monoLayer.monoParts[self.d].Evaluate(x)
            #print ('r: ', r, r.shape)
            #print ('x: ', x, x.shape)
        dr = tf.slice(g.gradient(r, x), [0,self.d],[numSamps,1])
        #print ('dr: ', dr)
        return tf.reduce_sum((0.5*tf.square(r) - tf.log(dr)))/float(numSamps)

bins = 20
n = 5000
lr = 0.01
opt = tf.train.AdamOptimizer(learning_rate=lr)
for i in range(200):
    opt.minimize(GaussianKL(0), var_list=monoLayer.trainable_variables)
    print('Dimension 0, Iteration %04d, Objective %04f'%(i,GaussianKL(0)().numpy()))
    if i % 50 == 0:
        r0 = monoLayer.genParts[0].Evaluate(x[:,:1]) + monoLayer.monoParts[0].Evaluate(x[:,:1])
        #print ('r0: ', r0.numpy().ravel().shape)
        #binwidth = 0.3
        plt.figure()
        data = r0.numpy().ravel()
        plt.hist(data, bins=np.linspace(min(data), max(data), bins))
        plt.title('Dimension 0 Mapped')
        plt.pause(0.05)
plt.show()
r0 = monoLayer.genParts[0].Evaluate(x[:,:1]) + monoLayer.monoParts[0].Evaluate(x[:,:1])

lr = 0.01
opt = tf.train.AdamOptimizer(learning_rate=lr)
for i in range(700):
    opt.minimize(GaussianKL(1), var_list=monoLayer.trainable_variables)
    r1 = monoLayer.genParts[1].Evaluate(x[:,:2]) + monoLayer.monoParts[1].Evaluate(x[:,:2])
    if i % 50 == 0: 
        plt.figure()   
        plt.scatter(r0.numpy().ravel(),r1.numpy().ravel(),alpha=0.2)
        plt.xlabel('r_1')
        plt.ylabel('r_2')
        plt.xlim([-4,4])
        plt.ylim([-4,4])
        plt.axis('equal')
        plt.title('Mapped Reference Samples')
        plt.pause(0.05)
    print('Dimension 1, Iteration %04d, Objective %04f'%(i,GaussianKL(1)().numpy()))
plt.show()
'''
opt = tf.train.AdamOptimizer(learning_rate=lr)
for i in range(n):
    opt.minimize(GaussianKL(2), var_list=monoLayer.trainable_variables)
    if (GaussianKL(2)().numpy() < 0):
        print ('Error minimized to 0. Continuing to next dimension.')
        break
    print('Dimension 2, Iteration %04d, Objective %04f'%(i,GaussianKL(2)().numpy()))
'''
print ('check x shape: ',x.shape)
r0 = monoLayer.genParts[0].Evaluate(x[:,:1]) + monoLayer.monoParts[0].Evaluate(x[:,:1])
r1 = monoLayer.genParts[1].Evaluate(x[:,:2]) + monoLayer.monoParts[1].Evaluate(x[:,:2])
#r2 = monoLayer.genParts[2].Evaluate(x[:,:3]) + monoLayer.monoParts[2].Evaluate(x[:,:3])


print ('r0: ', r0.shape)
print ('r1: ', r1.shape)
#print ('r2: ', r2.shape)

plt.scatter(xnump[:,0],xnump[:,1],alpha=0.2)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.title('Target Samples')
plt.show()
'''

fig = plt.figure()
ax = plt.axes(projection='3d')
zdata = xnump[:,0]
xdata = xnump[:,1]
ydata = xnump[:,2]
ax.scatter3D(xdata, ydata, zdata, c=zdata.ravel())
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('x_3')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_zlim([-4,4])
ax.set_title('Target Samples')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
zdata = r0.numpy().ravel()
xdata = r1.numpy().ravel()
ydata = r2.numpy().ravel()
ax.scatter3D(xdata, ydata, zdata, c=zdata)
ax.set_xlabel('r_1')
ax.set_ylabel('r_2')
ax.set_zlabel('r_3')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_zlim([-4,4])
ax.set_title('Mapped Reference Samples')
plt.show()
'''
plt.figure()
plt.scatter(r0.numpy().ravel(),r1.numpy().ravel(),alpha=0.2)
plt.xlabel('r_1')
plt.ylabel('r_2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.axis('equal')
plt.title('Mapped Reference Samples')


plt.show()