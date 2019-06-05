
#---Imports--------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import genData
import sys
from tensorflow.python.ops.parallel_for.gradients import jacobian

#---Global Variables-----------
lr = 1e-3
batchSize = 1
totalSamps = 10
numSamps = int(totalSamps/2)
epochNum = 10
Dim = 2
numQuad = 20
localDim = 2
#---Single Dimension Model----------------------

x = tf.placeholder("float", shape=[None,Dim], name='x')
#localDim = tf.placeholder(tf.int32, name='localDim')

#Set up quadrature
#numQuad = 20
quadPts = np.linspace(0,1,numQuad+1)[0:numQuad]
quadWts = (quadPts[1]-quadPts[0])*np.ones(numQuad)
#S = g + tf.reduce_sum(quadWts*x[:,-1]*tf.exp(h))

# Naive dense evaluation
#localx = x[:,:localDim] #not quite right

def mono():
	#Initialize learning parameters
	nCoef = 4 # TBD with smart way later, g() and h() could be different
	np.random.seed(0)
	init_w1 = np.random.rand(1, nCoef).astype(np.float32) 
	init_b1 = np.random.rand(nCoef).astype(np.float32)
	init_w2 = np.random.rand(1, nCoef).astype(np.float32) 
	#init_b2 = np.random.rand(nCoef).astype(np.float32)
	init_w3 = np.random.rand(nCoef, nCoef).astype(np.float32) 
	init_b3 = np.random.rand(nCoef).astype(np.float32)
	init_w4 = np.random.rand(nCoef, 1).astype(np.float32) 
	init_b4 = np.random.rand(1).astype(np.float32)
	print ('init_w1: ', init_w1)
	#init_b1 = np.zeros(nCoef)
	#function h() fully connected layers
	w1_1 = tf.get_variable(name='w1_1',dtype=tf.float32, initializer=init_w1)
	b1_1 = tf.get_variable(name='b1_1',dtype=tf.float32,initializer=init_b1)

	w1_2 = tf.get_variable(name='w1_2',dtype=tf.float32, initializer=init_w2)
	#b1_2 = tf.get_variable(name='b1_2', shape=[nCoef,1], dtype=tf.float32,trainable=True)

	w1_3 = tf.get_variable(name='w1_3',dtype=tf.float32,initializer=init_w3)
	b1_3 = tf.get_variable(name='b1_3',dtype=tf.float32,initializer=init_b3)

	w1_4 = tf.get_variable(name='w1_4',dtype=tf.float32,initializer=init_w4)
	b1_4 = tf.get_variable(name='b1_4',dtype=tf.float32,initializer=init_b4)

	compFirst = tf.reshape(x[:,0],[numSamps,1]) #tf.slice(x,[0,0], [numSamps,1]) #x[:,0]
	compLast = tf.reshape(x[:,-1], [numSamps,1]) #tf.slice(x,[0,localDim], [numSamps,1])  #x[:,localDim]
	print ('x: {}, compFirst: {}, compLast: {}'.format(x, compFirst, compLast))
	a = tf.matmul(compFirst, w1_1) + b1_1
	#print ('a: ', a)
	hOutput = 0.0
	for qIdx in range(numQuad):

		b1 = tf.nn.relu(tf.matmul(quadPts[qIdx]*compLast, w1_2))+a
		#print ('b1: ',b1)
		b2 = tf.nn.relu(tf.matmul(b1,w1_3)+b1_3)
		#print ('b2: ',b2)
		f = tf.matmul(b2, w1_4)+b1_4
		#print ('f: ', f)
		h = tf.square(f)
		#print ('h: ', h)
		hOutput += quadWts[qIdx]*h
	#print ('hOutput: {}, compLast: {}'.format(hOutput, compLast))
	hOutput = tf.multiply(hOutput, compLast)
	return hOutput

#function g() fully connected layers
def general(localDim):
	nCoef = 4
	np.random.seed(1)
	init_w1 = np.random.rand(localDim, nCoef).astype(np.float32)
	init_b1 = np.random.rand(nCoef).astype(np.float32)
	init_w2 = np.random.rand(nCoef, nCoef).astype(np.float32)
	init_b2 = np.random.rand(nCoef).astype(np.float32)
	init_w3 = np.random.rand(nCoef,1).astype(np.float32)
	init_b3 = np.random.rand(1).astype(np.float32)

	w2_1 = tf.get_variable(name='w2_1',dtype=tf.float32,initializer=init_w1)
	b2_1 = tf.get_variable(name='b2_1',dtype=tf.float32,initializer=init_b1)

	w2_2 = tf.get_variable(name='w2_2',dtype=tf.float32,initializer=init_w2)
	b2_2 = tf.get_variable(name='b2_2',dtype=tf.float32,initializer=init_b2)

	w2_3 = tf.get_variable(name='w2_3',dtype=tf.float32,initializer=init_w3)
	b2_3 = tf.get_variable(name='b2_3',dtype=tf.float32,initializer=init_b3)

	c1 = tf.nn.relu(tf.matmul(x[:,:localDim],w2_1)+b2_1)
	#print ('x: {}, c1: {}'.format(x[:,:localDim], c1))
	c2 = tf.nn.relu(tf.matmul(c1,w2_2)+b2_2)
	#print (c2, w2_3, b2_3)
	gOutput = tf.matmul(c2, w2_3)+b2_3
	return gOutput

S = general(localDim) + mono()
#print ('S: ', S, ' g: ', gOutput, ' h: ', hOutput, ' x: ', x)
#---Placeholders/Fnc Calls-----
J = tf.gradients(S,x)
print ('J: ', J[0])
dr = J[0][:,localDim-1]
loss = tf.reduce_sum((0.5*tf.square(S) - tf.log(dr)))/float(numSamps)
print ('loss: ', loss)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train = optimizer.minimize(loss)

# Run graph
def Train():
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	data = genData.Banana(totalSamps)
	dim = data.shape[1]
	print ('hello')
	for var in tf.trainable_variables():
		var_val = sess.run(var)
		print ('{}: {}'.format(var.name, var_val))
	for n in range(epochNum):
		Jac1 = sess.run(J, {x: data})
		Jac2 = sess.run(dr, {x: data})
		S = sess.run(S, {x: data})
		sess.run(train, {x: data})
		trainError = sess.run(loss, {x: data})
		#for var in tf.trainable_variables():
			#var_val = sess.run(var)
			#print ('{}: {}'.format(var.name, var_val))
		#print ('S: ', Sac)
		#print ('jacobian1: ',Jac1, Jac1[0][:,0])
		#print ('jacobian2: ', Jac2)
		print('Epoch #: {}, Dim: {}, Error: {}'.format(n, dim, trainError))

	#r0 = general(1)+mono()
	#r1 = general(2)+mono()
Train()



