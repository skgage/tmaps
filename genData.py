import numpy as np
import matplotlib.pyplot as plt

def Banana(numSamps):
	np.random.seed(0)
	numSamps = numSamps
	halfNumSamps = int(0.5*numSamps)
	x1a = np.random.randn(halfNumSamps,1).astype('f')
	x1b = np.random.randn(halfNumSamps,1).astype('f') + 2

	x2a = np.cos(x1a) + 0.3*np.random.randn(halfNumSamps,1).astype('f')
	x2b = np.cos(x1b) - 2.0 + 0.2*np.random.randn(halfNumSamps,1).astype('f')

	x1 = x1a;#np.concatenate([x1a,x1b])
	x2 = x2a;#np.concatenate([x2a,x2b])
	xnump = np.hstack([x1.reshape(-1,1),x2.reshape(-1,1)])

	#plt.plot(x1, x2, 'k.')
	#plt.show()
	return xnump

Banana(10)
