
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
    def __init__(self,d,layerSizes_s1, layerSizes_t1, layerSizes_s2, layerSizes_t2, monoLayer):
        
        self.d = d
        self.layerSizes_s1 = layerSizes_s1
        self.layerSizes_t1 = layerSizes_t1
        self.layerSizes_s2 = layerSizes_s2
        self.layerSizes_t2 = layerSizes_t2
        #self.groupSizes = groupSizes

        #self.ws = monoLayer.add_variable('g_ws_%d_0'%d, shape=[self.d+1, layerSizes[0]])
        #self.bs = monoLayer.add_variable('g_bs_%d_0'%d, shape=[layerSizes[0]])

        self.ws_first1 = monoLayer.add_variable('g_ws_first1', shape=[1])
        self.ts_first1 = monoLayer.add_variable('g_ts_first1', shape=[1])
        self.ws_first2 = monoLayer.add_variable('g_ws_first2', shape=[1])
        self.ts_first2 = monoLayer.add_variable('g_ts_first2', shape=[1])

        #for all components except the last one
        #NETWORK S1
        if self.d > 0:
            numLayers = len(layerSizes_s1)
            self.ws_s1 = [None]*(numLayers+1)
            self.bs_s1 = [None]*(numLayers+1)

            self.ws_s1[0] = monoLayer.add_variable('g_ws_s1_%d_0'%d, shape=[self.d,layerSizes_s1[0]])
            self.bs_s1[0] = monoLayer.add_variable('g_bs_s1_%d_0'%d, shape=[layerSizes_s1[0]])

            i = 0
            for i in range(1,numLayers):
                self.ws_s1[i] = monoLayer.add_variable('g_ws_s1_%d_%d'%(d,i),shape=[layerSizes_s1[i-1],layerSizes_s1[i]])
                self.bs_s1[i] = monoLayer.add_variable('g_bs_s1_%d_%i'%(d,i), shape=[layerSizes_s1[i]])

            self.ws_s1[i+1] = monoLayer.add_variable('g_ws_s1%d_%d'%(d,i+1), shape=[layerSizes_s1[i],self.d])
            self.bs_s1[i+1] = monoLayer.add_variable('g_bs_s1%d_%d'%(d,i+1), shape=[self.d])

            #NETWORK T1
            numLayers = len(layerSizes_t1)
            self.ws_t1 = [None]*(numLayers+1)
            self.bs_t1 = [None]*(numLayers+1)

            self.ws_t1[0] = monoLayer.add_variable('g_ws_t1_%d_0'%d, shape=[self.d,layerSizes_t1[0]])
            self.bs_t1[0] = monoLayer.add_variable('g_bs_t1_%d_0'%d, shape=[layerSizes_t1[0]])

            i = 0
            for i in range(1,numLayers):
                self.ws_t1[i] = monoLayer.add_variable('g_ws_t1_%d_%d'%(d,i),shape=[layerSizes_t1[i-1],layerSizes_t1[i]])
                self.bs_t1[i] = monoLayer.add_variable('g_bs_t1_%d_%i'%(d,i), shape=[layerSizes_t1[i]])

            self.ws_t1[i+1] = monoLayer.add_variable('g_ws_t1_%d_%d'%(d,i+1), shape=[layerSizes_t1[i],self.d])
            self.bs_t1[i+1] = monoLayer.add_variable('g_bs_t1_%d_%d'%(d,i+1), shape=[self.d])


            #NETWORK S2
            numLayers = len(layerSizes_s2)
            self.ws_s2 = [None]*(numLayers+1)
            self.bs_s2 = [None]*(numLayers+1)

            self.ws_s2[0] = monoLayer.add_variable('g_ws_s2_%d_0'%d, shape=[self.d,layerSizes_s2[0]])
            self.bs_s2[0] = monoLayer.add_variable('g_bs_s2_%d_0'%d, shape=[layerSizes_s2[0]])

            i = 0
            for i in range(1,numLayers):
                self.ws_s2[i] = monoLayer.add_variable('g_ws_s2_%d_%d'%(d,i),shape=[layerSizes_s2[i-1],layerSizes_s2[i]])
                self.bs_s2[i] = monoLayer.add_variable('g_bs_s2_%d_%i'%(d,i), shape=[layerSizes_s2[i]])

            self.ws_s2[i+1] = monoLayer.add_variable('g_ws_s2_%d_%d'%(d,i+1), shape=[layerSizes_s2[i],self.d])
            self.bs_s2[i+1] = monoLayer.add_variable('g_bs_s2_%d_%d'%(d,i+1), shape=[self.d])

            #NETWORK T2
            numLayers = len(layerSizes_t2)
            self.ws_t2 = [None]*(numLayers+1)
            self.bs_t2 = [None]*(numLayers+1)

            self.ws_t2[0] = monoLayer.add_variable('g_ws_t2_%d_0'%d, shape=[self.d,layerSizes_t2[0]])
            self.bs_t2[0] = monoLayer.add_variable('g_bs_t2_%d_0'%d, shape=[layerSizes_t2[0]])

            i = 0
            for i in range(1,numLayers):
                self.ws_t2[i] = monoLayer.add_variable('g_ws_t2_%d_%d'%(d,i),shape=[layerSizes_t2[i-1],layerSizes_t2[i]])
                self.bs_t2[i] = monoLayer.add_variable('g_bs_t2_%d_%i'%(d,i), shape=[layerSizes_t2[i]])

            self.ws_t2[i+1] = monoLayer.add_variable('g_ws_t2_%d_%d'%(d,i+1), shape=[layerSizes_t2[i],self.d])
            self.bs_t2[i+1] = monoLayer.add_variable('g_bs_t2_%d_%d'%(d,i+1), shape=[self.d])


        #will concatenate the output from this layer with last component then continue to last layer
        #layer for including monotone last component
        #self.wlast = monoLayer.add_variable('wlast_%d'%d, shape=[self.d+1,layerSizes[-1]])
        #self.blast = monoLayer.add_variable('blast_%d'%d, shape=[layerSizes[-1]])


    def Evaluate(self, x):
        x_mono = x
        if self.d != 0:
            x_nonmono = x[:,:-1] #all components except the last component
            #print ('x_nonmono: ', x_nonmono)
            x_first = x[:,0]
            #print ('x_nonmono: {}, self.d: {}'.format(x_nonmono.shape,self.d))
            x_lastcomp = tf.reshape(x[:,-1], [-1, 1])
            #print ('x_lastcomp: ', x_lastcomp.shape)
            #plt.figure()
            #plt.hist(x_lastcomp, bins='auto')
            #plt.show()
            numLayers = len(self.ws_s1)
            xLayers1 = [None]*numLayers
            #print ('ws_s1[0]: ', self.ws_s1[0], tf.reduce_sum(self.ws_s1[0]))
            xLayers1[0] = tf.nn.relu(tf.matmul(x_nonmono, self.ws_s1[0]) + self.bs_s1[0])
            for i in range(1,numLayers):
                xLayers1[i] = tf.matmul(xLayers1[i-1], self.ws_s1[i]) + self.bs_s1[i]
                #print ('ws_s1[{}]: {}'.format(i, self.ws_s1[i]), tf.reduce_sum(self.ws_s1[i]))
                if i != numLayers - 1:
                    xLayers1[i] = tf.nn.relu(xLayers1[i])
            s1_out = tf.square(xLayers1[-1])
            #print ('s1_out: ', s1_out, xLayers1[i])
            numLayers = len(self.ws_t1)
            xLayers2 = [None]*numLayers
            xLayers2[0] = tf.nn.relu(tf.matmul(x_nonmono, self.ws_t1[0]) + self.bs_t1[0])
            #print ('ws_t1[0]: ', self.ws_t1[0], tf.reduce_sum(self.ws_t1[0]))
            for i in range(1,numLayers):
                #print (i, ' of numLayers: ', numLayers)
                xLayers2[i] = tf.matmul(xLayers2[i-1], self.ws_t1[i]) + self.bs_t1[i]
                #print ('ws_t1[{}]: {}'.format(i, self.ws_t1[i]), tf.reduce_sum(self.ws_t1[i]))
                if i != numLayers - 1:
                    xLayers2[i] = tf.nn.relu(xLayers2[i])
            t1_out = xLayers2[-1]
            #print ('t1_out: ', t1_out, xLayers2[i])

            y1_1 = x_nonmono*tf.exp(self.ws_first1) + self.ts_first1 #first layer output components
            #print ('x_lastcomp: ', x_lastcomp.shape)
            y2_1 = x_lastcomp*s1_out + t1_out
            #print ('y2_1: ', y2_1)
            
            x_nonmono = tf.reshape(y1_1, [-1,1])
            x_lastcomp = tf.reshape(y2_1, [-1,1])
            #Now to compute coupled layer outputs
            numLayers = len(self.ws_s2)
            xLayers3 = [None]*numLayers
            xLayers3[0] = tf.matmul(x_nonmono, self.ws_s2[0]) + self.bs_s2[0]
            for i in range(1,numLayers):
                xLayers3[i] = tf.matmul(xLayers3[i-1], self.ws_s2[i]) + self.bs_s2[i]
            s2_out = tf.square(xLayers3[-1])
            #print ('s2: ', s2_out)
            numLayers = len(self.ws_t2)
            xLayers4 = [None]*numLayers
            xLayers4[0] = tf.matmul(x_nonmono, self.ws_t2[0]) + self.bs_t2[0]
            for i in range(1,numLayers):
                xLayers4[i] = tf.matmul(xLayers4[i-1], self.ws_t2[i]) + self.bs_t2[i]
            t2_out = xLayers4[-1]
            #print ('t2: ', t2_out)
            #y1_2 = x_nonmono*np.exp(self.ws_first2) + self.ts_first2 #second layer output components
            y2_2 = x_lastcomp*s2_out + t2_out
            #print ('y2_2: ', y2_2)
            
            return y2_2 + 0.000001
            #print ('self.ws: ', self.ws)
            #print ('self.bs: ', self.bs)
            #xLayers[-1] to be concatenated with the the last/monotone component
            #print ('xLayer[-1]: {} x_lastcomp: {}'.format(xLayers[-1].shape, x_lastcomp.shape))
        else:
            y1_1 = x*tf.exp(self.ws_first1) + self.ts_first1
            y1_2 = y1_1*tf.exp(self.ws_first2) + self.ts_first2
            return y1_2

class MonotoneLayer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(MonotoneLayer, self).__init__()
        self.num_outputs = dim
        self.dim = dim

        layerSizes_s1 = [8,8,8] 
        layerSizes_t1 = [8,8,8]
        layerSizes_s2 = [8,8,8]
        layerSizes_t2 = [8,8,8]
        #groupSizes = [2,2,2,2] #only used for the min-max layer
        #assert (np.sum(groupSizes) == layerSizes[-1]),"Group sizes must sum to layer size."
        #self.monoParts = [ MonotoneDimension(d, layerSizes, self) for d in range(dim)]
        self.genParts = [ GeneralDimension(d, layerSizes_s1, layerSizes_t1, layerSizes_s2, layerSizes_t2,  self) for d in range(dim)]
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
numSamps = 8000
halfNumSamps = int(0.5*numSamps)
x1a = np.random.randn(halfNumSamps,1).astype('f')
x1b = np.random.randn(halfNumSamps,1).astype('f') + 2

x2a = np.cos(x1a) + 0.3*np.random.randn(halfNumSamps,1).astype('f')
x2b = np.cos(x1b) - 2.0 + 0.2*np.random.randn(halfNumSamps,1).astype('f')

a = np.random.normal(10,1,int(0.5*halfNumSamps))
b = np.random.normal(5,2,int(0.5*halfNumSamps))
c = np.hstack([a.reshape(-1,1),b.reshape(-1,1)])
x1 = x1a #c.astype('float32');#np.concatenate([x1a,x1b])
x2 = x2a;#np.concatenate([x2a,x2b])
#x1 = x2a
#x2 = c.astype('float32')
xnump = np.hstack([x1.reshape(-1,1),x2.reshape(-1,1)])

x = tf.constant(xnump)

plt.scatter(xnump[:,0],xnump[:,1],alpha=0.1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Original dataset')
plt.show()
# quit()
print ('xnump: ', xnump.shape)
#xnump = genData.Banana(halfNumSamps)
#print ('xnump PICK ME: ', xnump)
#x = tf.constant(xnump)
plt.figure()
plt.hist(xnump[:,0], bins='auto')
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

def grad_logging(dr):
    dr_log = tf.log(dr)
    dr_log_nonan = dr_log
    dr_log_nonan = tf.where(tf.is_nan(dr_log), tf.zeros_like(dr_log), dr_log)
    dr_log_nonan = tf.where(tf.is_inf(tf.math.abs(dr_log)), tf.zeros_like(dr_log_nonan), dr_log_nonan)
    return dr_log_nonan

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
            r0 = monoLayer.genParts[0].Evaluate(x[:,:1])
            #print ('r0: ', r0, r0.shape)
            grad = g.gradient(r0,x)
            print ('dr0: ', grad)
            dr0 = tf.slice(grad, [0,0],[numSamps,1])

        with tf.GradientTape() as h:  
            h.watch(x)  
            r1 = monoLayer.genParts[1].Evaluate(x)
            #print ('r1: ', r1, r1.shape)
            #print ('x: ', x, x.shape)
            grad = h.gradient(r1,x)
            #print ('dr1: ', grad)
            dr1 = tf.slice(grad, [0,1],[numSamps,1])
            #dr = g.gradient(r, x)
        #r = monoLayer.genParts[self.d].Evaluate(x) #[:,:self.d+1])
        #print ('r: ', r, r.shape)
        #print ('gradient before slicing: ',g.gradient(r,x))
        #print ('x: ', x)
        
        #dr = tf.slice(r/x, [0,self.d],[numSamps,1])
        
        #print ('second dr test: ', tf.gradients(r,x))
        #dr = g.gradient(r, x)
        #print ('dr: ', dr.numpy(), tf.reduce_sum(dr))
        #dr_log = tf.log(dr)
        #print ('dr_log: ', dr_log)
        #dr_log_nonan = dr_log
        #dr_log_nonan = tf.where(tf.is_nan(dr_log), tf.zeros_like(dr_log), dr_log)
        #dr_log_nonan = tf.where(tf.is_inf(tf.math.abs(dr_log)), tf.zeros_like(dr_log_nonan), dr_log_nonan)
        #print ('dr_log_nonan: ', dr_log_nonan)
        dr_log0 = grad_logging(dr0)
        dr_log1 = grad_logging(dr1)
        return tf.reduce_sum((0.5*tf.square(r0) - dr_log0))/float(numSamps) + tf.reduce_sum((0.5*tf.square(r1) - dr_log1))/float(numSamps)

#def animate(i):

bins = 20
#plt.figure()
#fig, (ax1) = plt.subplots(nrows=1)
#line1, = ax1.scatter(xnump[:,0],xnump[:,1])
n = 7000
'''
lr = 0.1
opt = tf.train.AdamOptimizer(learning_rate=lr)
for i in range(1000):
    opt.minimize(GaussianKL(0), var_list=monoLayer.trainable_variables)
    #if (GaussianKL(0)().numpy() < 0):
        #print ('Error minimized to 0. Continuing to next dimension.')
        #break
    print('Dimension 0, Iteration %04d, Objective %04f'%(i,GaussianKL(0)().numpy()))
    if i % 100 == 0:
        r0 = monoLayer.genParts[0].Evaluate(np.reshape(xnump[:,0], [halfNumSamps, 1]))
        print ('r0: ', r0.shape)

        plt.figure()
        data = r0.numpy().ravel()
        plt.hist(data, bins=np.linspace(min(data), max(data), bins))
        plt.title('Dimension 0 Mapped')
        plt.pause(0.05)
plt.show()
'''
#plt.ion()
lr = 0.0001
opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=False)
for i in range(30000):
    opt.minimize(GaussianKL(1), var_list=monoLayer.trainable_variables)
    #if i == 0:
        #start = monoLayer.trainable_variables
    #if (GaussianKL(1)().numpy() < 0):
        #print ('Error minimized to 0. Continuing to next dimension.')
        #break
    #for var in monoLayer.trainable_variables:
        #print (var)
    print('Dimension 1, Iteration %04d, Objective %04f'%(i,GaussianKL(1)().numpy()))
    if (i % 200 == 0):
        r0 = monoLayer.genParts[0].Evaluate(xnump[:,0])
        r1 = monoLayer.genParts[1].Evaluate(xnump)
        plt.figure()
        plt.scatter(r0.numpy().ravel(),r1.numpy().ravel(),alpha=0.2)
        plt.xlabel('r_1')
        plt.ylabel('r_2')
        plt.xlim([-4,4])
        plt.ylim([-4,4])
        plt.axis('equal')
        plt.title('Mapped Reference Samples')
        '''
        plt.figure()
        data = r1.numpy().ravel()
        plt.hist(data, bins=np.linspace(min(data), max(data), bins))
        plt.title('Dimension 1 Mapped')
        '''
        #plt.show()
        plt.pause(0.05)
        
plt.show()

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