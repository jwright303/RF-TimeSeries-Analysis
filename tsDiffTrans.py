import numpy as np
import matplotlib.pyplot as plt

#####################################
# Variables:
#
# 	Window Size: Nw = 100
#	Magnitude Threshold: Lw = 0.005
#	Minimum Packet Size: Npkt = 1000
#	Maximum Ack Size: Nack = 2000
#
#####################################

NW = 100
#LW = 0.02
LW = 0.005
NPKT = 1000
NACK = 2000
PREV = 2
filename='bigdata.bin'


# Broken down into two parts - first signal detection then packet detection

if __name__ == "__main__":
	# Indexes for the window that we are currently look at - a subset of the total samples
	startIdx = 0
	endIdx = NW
	path = "/Volumes/Jack_SSD/Outdoor/Day_4/Device_12/"
	filename1 = path + "tx_3_iq.dat"
	filename2 = path + "tx_4_iq.dat"

	rfFile = np.fromfile(filename1, dtype=np.float32)
	rfFile2 = np.fromfile(filename2, dtype=np.float32)

	# Square all values to help get the power

	rfFile = np.square(rfFile)
	rfFile2 = np.square(rfFile2)
	smpls = len(rfFile)//2

	# Splits the IQ array into arrays of the I and Q values seperately
	realV = rfFile[::2]
	complexV = rfFile[1::2]

	realV2 = rfFile2[::2]
	complexV2 = rfFile2[1::2]	

	# The number of times we will look at a window of samples - this takes the ceiling of samples divided by the window size
	numWindows = (-1 * (-smpls // NW))
	
	sigs = []
	sigs2 = []

	# Iterate through the values in the rf File - one window at at time - note that every other entry will be the complex value corresponding to the IQ sample
	for i in range(0, numWindows):
		curReals = realV[startIdx : endIdx]
		curComp = complexV[startIdx : endIdx]

		curReals2 = realV2[startIdx : endIdx]
		curComp2 = complexV2[startIdx : endIdx]


		# Since already squared this is I^2 + Q^2
		# Sum in the window
		mag = np.sum(curReals + curComp)
		mag2 = np.sum(curReals2 + curComp2)

		sigs.append(mag)
		sigs2.append(mag2)

		startIdx = endIdx
		if endIdx + NW > smpls:
			endIdx += (smpls - startIdx)
		else:
			endIdx += NW

	
	# y2 = sigs[:-PREV]
	# y = sigs[PREV:]	
	#print(len(y2), len(y))
	plt.scatter(sigs, sigs2, s=1, alpha=1)
	#plt.plot(x, a*x+b)
	plt.show()



