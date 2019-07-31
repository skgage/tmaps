import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import invgauss

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
	

	#fig = plt.figure()
	#ax = plt.axes(projection='3d')
	zdata = x1a
	xdata = x2a
	ydata = x1a
	#ax.scatter3D(xdata, ydata, zdata, c=zdata.ravel())
	#plt.show()
	#print (xdata,ydata,zdata)
	xnump2 = np.hstack([xdata.reshape(-1,1),ydata.reshape(-1,1),zdata.reshape(-1,1)])

	test = x2b
	#plt.figure()
	#plt.hist(test, bins='auto')
	#plt.show()
	print (xnump.shape)
	return xnump


Banana(8000)
