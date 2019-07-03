
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import genData
import matplotlib.animation

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

tf.enable_eager_execution()

class GeneralDimension:
    def __init__(self,d,layerSizes,groupSizes, monoLayer):
        numLayers = len(layerSizes)
        self.d = d
        self.layerSizes = layerSizes
        self.groupSizes = groupSizes

        #self.ws = monoLayer.add_variable('g_ws_%d_0'%d, shape=[self.d+1, layerSizes[0]])
        #self.bs = monoLayer.add_variable('g_bs_%d_0'%d, shape=[layerSizes[0]])

        #for all components except the last one
        self.ws = [None]*(numLayers)
        self.bs = [None]*(numLayers)

        self.ws[0] = monoLayer.add_variable('g_ws_%d_0'%d, shape=[self.d,layerSizes[0]])
        self.bs[0] = monoLayer.add_variable('g_bs_%d_0'%d, shape=[layerSizes[0]])

        i = 0
        for i in range(1,numLayers-1):
            self.ws[i] = monoLayer.add_variable('g_ws_%d_%d'%(d,i),shape=[layerSizes[i-1],layerSizes[i]])
            self.bs[i] = monoLayer.add_variable('g_bs_%d_%i'%(d,i), shape=[layerSizes[i]])

        self.ws[i+1] = monoLayer.add_variable('g_ws_%d_%d'%(d,i+1), shape=[layerSizes[i],self.d])
        self.bs[i+1] = monoLayer.add_variable('g_bs_%d_%d'%(d,i+1), shape=[self.d])
        #will concatenate the output from this layer with last component then continue to last layer
        #layer for including monotone last component
        self.wlast = monoLayer.add_variable('wlast_%d'%d, shape=[self.d+1,layerSizes[-1]])
        self.blast = monoLayer.add_variable('blast_%d'%d, shape=[layerSizes[-1]])

    def Evaluate(self, x):
        x_mono = x
        if self.d != 0:
            x_nonmono = x[:,:-1] #all components except the last component
            #print ('x_nonmono: {}, self.d: {}'.format(x_nonmono.shape,self.d))
            x_lastcomp = np.reshape(x[:,-1], [-1, 1])
            numLayers = len(self.ws)
            xLayers = [None]*numLayers
            xLayers[0] = tf.nn.relu(tf.matmul(x_nonmono, self.ws[0]) + self.bs[0])
            for i in range(1,numLayers):
                xLayers[i] = tf.nn.relu(tf.matmul(xLayers[i-1], self.ws[i]) + self.bs[i])
            #print ('self.ws: ', self.ws)
            #print ('self.bs: ', self.bs)
            #xLayers[-1] to be concatenated with the the last/monotone component
            #print ('xLayer[-1]: {} x_lastcomp: {}'.format(xLayers[-1].shape, x_lastcomp.shape))
            x_mono = tf.reshape(tf.stack([xLayers[-1], x_lastcomp]), [-1, self.d+1]) #input to monolayer    
        
        monoVar = tf.reshape(np.exp((self.wlast[-1,:])), [1, self.layerSizes[-1]])
        #print ('other weights: {}, mono part: {}'.format(self.wlast[:-1,:].shape, monoVar))
        if self.d != 0:
            wmono = tf.stack([self.wlast[:-1, :], monoVar])
        else:
            wmono = monoVar
        #self.ws[-1,:] = monoVar 
        #x = np.reshape(x, [-1, self.d+1])
        wmono = np.reshape(wmono, [self.d+1, self.layerSizes[-1]])
        #print ('weight matrix: ', w)
        #print ('weight matrix: ', w)     
        #print ('bias matrix: ', self.bs)     
        #print ('x_mono: {}, wmono: {}'.format(x_mono.shape, wmono.shape))                              
        xlayer = tf.matmul(x_mono, wmono) + self.blast
        #print ('xlayer: ', xlayer)
        groupsIdx = [0]
        for i in range(len(self.groupSizes)):
            groupsIdx.append(groupsIdx[i]+self.groupSizes[i])
        #print ('group indices: ', groupsIdx)

        groupsMin = [] #g1, g2, and so on
        for i in range(len(self.groupSizes)):
            group = xlayer[:,groupsIdx[i]:groupsIdx[i+1]]
            #print ('group: ', i, ' ', group)
            gMin = tf.math.reduce_min(group, axis = 1)
            #print ('gMin: ', gMin)
            gMin = tf.reshape(gMin, [-1, 1])
            groupsMin.append(gMin)
        #print ('groupsMin: ', groupsMin)
        groupsMin = tf.reshape(tf.stack([groupsMin]), [-1, len(self.groupSizes)])
        #print ('groupsMin: ', groupsMin)
        #print ('stacked groupsMix: ', )
        Ox = tf.math.reduce_max(groupsMin, axis = 1)
        #print ('Ox: ', Ox)
        Ox = tf.reshape(Ox, [-1, 1])
        #Group1 = xlayer[:,:groups[0]]
        #Group2 = xlayer[:,groups[0]:]
        #print ('Group1: {}, Group2: {}'.format(Group1, Group2))
        #g1 = tf.reshape(tf.math.reduce_min(Group1, axis=1), [-1, self.d+1])
        #g2 = tf.reshape(tf.math.reduce_min(Group2, axis=1), [-1, self.d+1])
        #print ('g1: {}, g2: {}'.format(g1, g2))
        #Ox = tf.reshape(tf.math.reduce_max([tf.stack([g1, g2])], axis=1), [-1,self.d+1])
        #print ('Ox: ', Ox)
        return Ox

class MonotoneLayer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(MonotoneLayer, self).__init__()
        self.num_outputs = dim
        self.dim = dim

        layerSizes = [32,8] 
        groupSizes = [2,2,2,2] #only used for the min-max layer
        assert (np.sum(groupSizes) == layerSizes[-1]),"Group sizes must sum to layer size."
        #self.monoParts = [ MonotoneDimension(d, layerSizes, self) for d in range(dim)]
        self.genParts = [ GeneralDimension(d, layerSizes, groupSizes, self) for d in range(dim)]
        #print ('monoParts: ', self.monoParts)
    def build(self, input_shape):
        print('input_shape = ', input_shape)
        assert input_shape[-1] == self.num_outputs
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        print ('IS THIS HAPPENING??')
        #out = tf.concat([self.monoParts[d].Evaluate(input) + self.genParts[d].Evaluate(input) for d in range(self.dim)], 1)
        out = [self.genParts[d].Evaluate(input) for d in range(self.dim)]
        return out

np.random.seed(0)
numSamps = 4000
halfNumSamps = int(0.5*numSamps)
x1a = np.random.randn(halfNumSamps,1).astype('f')
x1b = np.random.randn(halfNumSamps,1).astype('f') + 2

x2a = np.cos(x1a) + 0.3*np.random.randn(halfNumSamps,1).astype('f')
x2b = np.cos(x1b) - 2.0 + 0.2*np.random.randn(halfNumSamps,1).astype('f')

x1 = x1a;#np.concatenate([x1a,x1b])
x2 = x2a;#np.concatenate([x2a,x2b])
xnump = np.hstack([x1.reshape(-1,1),x2.reshape(-1,1)])

x = tf.constant(xnump)

#plt.scatter(xnump[:,0],xnump[:,1],alpha=0.1)
#plt.show()
# quit()
print ('xnump: ', xnump.shape)
#xnump = genData.Banana(halfNumSamps)
#print ('xnump PICK ME: ', xnump)
#x = tf.constant(xnump)
bins = 20
data = xnump[:,0]
plt.figure()
plt.hist(xnump[:,0], bins=np.linspace(min(data), max(data), bins))
plt.show()
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
            #r = monoLayer.genParts[self.d].Evaluate(x[:,:self.d+1]) + monoLayer.monoParts[self.d].Evaluate(x[:,:self.d+1])
            r = monoLayer.genParts[self.d].Evaluate(x[:,:self.d+1])
            #print ('r: ', r, r.shape)
            #print ('x: ', x, x.shape)
        dr = tf.slice(g.gradient(r, x), [0,self.d],[numSamps,1])
        #print ('dr: ', dr)
        dr_log = tf.log(dr)
        #print ('dr_log: ', dr_log)
        dr_log_nonan = tf.where(tf.is_nan(dr_log), tf.zeros_like(dr_log), dr_log)
        dr_log_nonan = tf.where(tf.is_inf(tf.math.abs(dr_log)), tf.zeros_like(dr_log_nonan), dr_log_nonan)
        #print ('dr_log_nonan: ', dr_log_nonan)
        return tf.reduce_sum((0.5*tf.square(r) - dr_log_nonan))/float(numSamps)

#def animate(i):


plt.figure()
#fig, (ax1) = plt.subplots(nrows=1)
#line1, = ax1.scatter(xnump[:,0],xnump[:,1])
n = 7000
lr = 0.01
opt = tf.train.AdamOptimizer(learning_rate=lr)
for i in range(1000):
    opt.minimize(GaussianKL(0), var_list=monoLayer.trainable_variables)
    #if (GaussianKL(0)().numpy() < 0):
        #print ('Error minimized to 0. Continuing to next dimension.')
        #break
    print('Dimension 0, Iteration %04d, Objective %04f'%(i,GaussianKL(0)().numpy()))
r0 = monoLayer.genParts[0].Evaluate(np.reshape(xnump[:,0], [halfNumSamps, 1]))
print ('r0: ', r0.numpy().ravel().shape)
#binwidth = 0.3
plt.figure()
data = r0.numpy().ravel()
plt.hist(data, bins=np.linspace(min(data), max(data), bins))
plt.title('Dimension 0 Mapped')
plt.show()
#plt.ion()
opt = tf.train.AdamOptimizer(learning_rate=lr)
for i in range(1000):
    opt.minimize(GaussianKL(1), var_list=monoLayer.trainable_variables)
    #if (GaussianKL(1)().numpy() < 0):
        #print ('Error minimized to 0. Continuing to next dimension.')
        #break
    print('Dimension 1, Iteration %04d, Objective %04f'%(i,GaussianKL(1)().numpy()))
r1 = monoLayer.genParts[1].Evaluate(np.reshape(xnump[:,:2], [halfNumSamps, 2]))
plt.figure()
data = r1.numpy().ravel()
plt.hist(data, bins=np.linspace(min(data), max(data), bins))
plt.title('Dimension 1 Mapped')
plt.show()
'''
plt.scatter(r0.numpy().ravel(),r1.numpy().ravel(),alpha=0.2)
plt.xlabel('r_1')
plt.ylabel('r_2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.axis('equal')
plt.title('Mapped Reference Samples')
#plt.show()
plt.pause(0.05)

#plt.ioff()
plt.show()
'''


'''
opt = tf.train.AdamOptimizer(learning_rate=lr)
for i in range(n):
    opt.minimize(GaussianKL(2), var_list=monoLayer.trainable_variables)
    if (GaussianKL(2)().numpy() < 0):
        print ('Error minimized to 0. Continuing to next dimension.')
        break
    print('Dimension 2, Iteration %04d, Objective %04f'%(i,GaussianKL(2)().numpy()))
'''
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


plt.figure()
plt.scatter(r0.numpy().ravel(),r1.numpy().ravel(),alpha=0.2)
plt.xlabel('r_1')
plt.ylabel('r_2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.axis('equal')
plt.title('Mapped Reference Samples')


plt.show()
'''