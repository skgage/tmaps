import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
        print ('ws: ', self.ws, ' bs: ', self.bs)
        # set up a left endpoint rule for the integration
        self.numQuad = 20
        self.quadPts = np.linspace(0,1,self.numQuad+1)[0:self.numQuad]
        self.quadWts = (self.quadPts[1]-self.quadPts[0])*np.ones(self.numQuad)
        print ('quad pts: ', self.quadPts, ' self.quadWts: ', self.quadWts)
        # Set up the weights
        self.wd = monoLayer.add_variable("wd_%d"%d, shape = [ 1,layerSizes[0]])
        print ("wd_%d"%d)
        if(d!=0):
            print ('size_0: [', d, ', ', layerSizes[0], ']')
            self.ws[0] = monoLayer.add_variable('ws_%d_0'%d, shape=[ d,layerSizes[0]])
            self.bs[0] = monoLayer.add_variable('bs_%d_0'%d, shape=[ layerSizes[0] ])

        for i in range(1,numLayers):
            print ('size_{}: [{}, {}]'.format(i, layerSizes[i-1], layerSizes[i]))
            self.ws[i] = monoLayer.add_variable('ws_%d_%d'%(d,i), shape=[ layerSizes[i-1], layerSizes[i] ])
            self.bs[i] = monoLayer.add_variable('ws_%d_%d'%(d,i), shape=[layerSizes[i]])

        # finally, combine everything into a scalar
        print ('size_{}: [{}, {}]'.format(i, layerSizes[i-1], 1))
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
        return tf.concat([self.monoParts[d].Evaluate(input) + self.genParts[d].Evaluate(input) for d in range(self.dim)], 1)

numSamps = 8000
halfNumSamps = int(0.5*numSamps)
x1a = np.random.randn(halfNumSamps,1).astype('f')
x1b = np.random.randn(halfNumSamps,1).astype('f') + 2

x2a = np.cos(x1a) + 0.3*np.random.randn(halfNumSamps,1).astype('f')
x2b = np.cos(x1b) - 2.0 + 0.2*np.random.randn(halfNumSamps,1).astype('f')

x1 = x1a;#np.concatenate([x1a,x1b])
x2 = x2a;#np.concatenate([x2a,x2b])
xnump = np.hstack([x1.reshape(-1,1),x2.reshape(-1,1)])

x = tf.constant(xnump)

# plt.scatter(xnump[:,0],xnump[:,1],alpha=0.1)
# plt.show()
# quit()
print ('xnump: ', xnump.shape)
monoLayer = MonotoneLayer(xnump.shape[1])

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
            r = monoLayer.genParts[self.d].Evaluate(x) + monoLayer.monoParts[self.d].Evaluate(x)
            #print ('r: ', r, r.shape)
            #print ('x: ', x, x.shape)
        dr = tf.slice(g.gradient(r, x), [0,self.d],[numSamps,1])
        return tf.reduce_sum((0.5*tf.square(r) - tf.log(dr)))/float(numSamps)


opt = tf.train.AdamOptimizer(learning_rate=0.01)
for i in range(1):
    opt.minimize(GaussianKL(0), var_list=monoLayer.trainable_variables)
    print('Dimension 0, Iteration %04d, Objective %04f'%(i,GaussianKL(0)().numpy()))

opt = tf.train.AdamOptimizer(learning_rate=0.01)
for i in range(1):
    opt.minimize(GaussianKL(1), var_list=monoLayer.trainable_variables)
    print('Dimension 1, Iteration %04d, Objective %04f'%(i,GaussianKL(1)().numpy()))

r0 = monoLayer.genParts[0].Evaluate(x) + monoLayer.monoParts[0].Evaluate(x)
r1 = monoLayer.genParts[1].Evaluate(x) + monoLayer.monoParts[1].Evaluate(x)
print ('r0: ', r0.shape)
print ('r1: ', r1.shape)
plt.scatter(xnump[:,0],xnump[:,1],alpha=0.2)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.title('Target Samples')

plt.figure()
plt.scatter(r0.numpy().ravel(),r1.numpy().ravel(),alpha=0.2)
plt.xlabel('r_1')
plt.ylabel('r_2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.axis('equal')
plt.title('Mapped Reference Samples')

plt.show()