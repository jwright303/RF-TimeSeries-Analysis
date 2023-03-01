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
PREV = 4
filename='bigdata.bin'


# Broken down into two parts - first signal detection then packet detection

if __name__ == "__main__":
	# Indexes for the window that we are currently look at - a subset of the total samples
	startIdx = 0
	endIdx = NW
	path = "/Volumes/Jack_SSD/Outdoor/Day_4/Device_12/"
	filename = path + "tx_1_iq.dat"

	rfFile = np.fromfile(filename, dtype=np.float32)

	# Square all values to help get the power

	rfFile = np.square(rfFile)
	smpls = len(rfFile)//2

	# Splits the IQ array into arrays of the I and Q values seperately
	realV = rfFile[::2]
	complexV = rfFile[1::2]

	# The number of times we will look at a window of samples - this takes the ceiling of samples divided by the window size
	numWindows = (-1 * (-smpls // NW))
	
	signals = np.array([0.0] * numWindows)
	pastSigs = np.array([0.0] * (numWindows - 2))
	mags = []
	sigs = []

	sigStart = 0
	sigCount = 0
	final = []

	# Iterate through the values in the rf File - one window at at time - note that every other entry will be the complex value corresponding to the IQ sample
	for i in range(0, numWindows):
		curReals = realV[startIdx : endIdx]
		curComp = complexV[startIdx : endIdx]

		# Since already squared this is I^2 + Q^2
		# Sum in the window
		mag = np.sum(curReals + curComp)
		mags.append(mag)
		if mag >= LW:
			sigs.append(mag)
			if sigCount == 0:
				sigStart = i
			sigCount += 1
		else:
			if sigCount >= NPKT:
				final += sigs[sigStart:i]
			sigCount = 0

		startIdx = endIdx
		if endIdx + NW > smpls:
			endIdx += (smpls - startIdx)
		else:
			endIdx += NW

	y2 = mags[:-PREV]
	y = mags[PREV:]
	print(len(y2), len(y))

	plt.scatter(y, y2, s=1, alpha=1)
	plt.show()



