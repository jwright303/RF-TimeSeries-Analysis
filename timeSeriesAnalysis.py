import timeSeriesHelper as tsHelper
import pandas as pd
import numpy as np
from scipy import stats as s
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#Path to the RF data
# path = "/Volumes/Jack_SSD/Outdoor/Day_4/Device_11/"
# name = "tx_3_iq.dat"

# Function to analyze some information about a packet. Takes in the packet magnitude values
# Gives a basic description of the packet, shows the periodogram value and the autocorrelation
# Returns nothing but displays valuable information
def analyzePacket(pack, res):
	plt.plot(pack)
	plt.show()

	pckDf = pd.DataFrame(pack, columns = ['Val'])
	tsHelper.getStationaryStats(pckDf)

	analyzePacketPer(pack, verbose=True)

	#print("max vals", Pxx_den[maxI], f[maxI])
	tsHelper.showPeriodogram(pckDf)
	tsHelper.showPlot()

	tsHelper.showAutoCorrellation(pckDf)
	tsHelper.showPlot()

# Function gets information about a packet periodogram value
# The function takes in the packet and the option to be verbose
# Returns the dominant cycle within the packet
def analyzePacketPer(pack, verbose=False):	
	pckDf = pd.DataFrame(pack, columns = ['Val'])

	#f is an array of sample requencies, and pxx is the power spectral density
	f, pxx = tsHelper.getPeriodogram(pckDf)
	perdArr = np.array(pxx)
	maxI = np.argmax(perdArr)

	if verbose == True:
		print("max vals", pxx[maxI], f[maxI])

	return f[maxI]

# Function to get all dominant cycles from a list of packets
# Function takes in list of packets, and prints out a dictionary containing the dominant cycle value and how many times it occurs
# Function returns the dicitionary and average periodogram value
def getPeridogramVals(packs, verbose=False):
	rSum = 0
	res = {}
	for pac in packs:
		v = analyzePacketPer(pac)

		key = int(v)
		if key in res.keys():
			res[key] += 1
		else:
			res[key] = 1
		rSum += v
	
	if verbose == True:
		print("Per Values", res)
		print("Average Per val:", rSum/float(len(packs)))

	return res, rSum/float(len(packs))

# Function to parse periodogram values from an earlier function
# Prints out a cleaned version of the output
def parsePeriodogramTransmissions(filename):
	f = open(filename, "r")
	for x in f:
		devCheck = x[:6]
		valCheck = x[:3]

		if devCheck == "Device":
			print("\n", x)
		elif valCheck == "Per":
			print(x, end="")
		elif valCheck == "Day":
			print("")

	f.close()

# # Function to parse periodogram values from an earlier function
# # returns the periodogram value corresponding to the dominant cycle
# def getTransmissionPeriodogramVal(transStr):
# 	dictArr = transStr.split(", ")

# 	keys = []
# 	vals = []

# 	for pair in dictArr:
# 		vs = pair.split(": ")
# 		keys.append(int(vs[0]))
# 		vals.append(int(vs[1]))

# 	ind = np.argmax(vals)
# 	#print(keys[ind])
# 	return keys[ind]

# Function to parse periodogram values from earlier function. The aim here is to get the most dominant periodogram value for a device for that day
# Function prints out the most dominant cycle for the day and returns nothing
def getValPerDay(filename):
	f = open(filename, "r")
	i = 0
	cur = []

	for x in f:
		res = x.split("{")

		if len(res) == 2:
			dictStr = (res[1].split("}"))[0]
			key = getTransmissionPeriodogramVal(dictStr)
			cur.append(key)
		
		else:
			if len(cur) > 0:
				finDev = (s.mode(cur)[0])
				print(finDev[0])
				#print("")
				cur.clear()
			v = res[0]
			if len(v) != 1:
				#print("")
				pass

	finDev = (s.mode(cur)[0])
	print(finDev[0])

	return

# Function offers several ways to visualize periodogram values for different days
# Returns nothing but displays different graphs
def graphPeriodogramDayVals(filename, hist=False, scatter=False, line=False, combinedLine=False):
	arr = np.fromfile(filename, sep='\n')
	print(len(arr))

	t1 = arr[::3]
	t2 = arr[1::3]
	t3 = arr[2::3]

	x = np.arange(len(t1))

	#print(t1)
	if hist == True:
		plt.hist(t3, bins=50, color='#fcba03')
		plt.title('Histogram of Periodogram Values for All Devices, One Transmission')
		plt.xlabel('Periodogram Value')
		plt.ylabel('Number of Occurances')
		plt.show()
	if scatter == True:
		plt.scatter(x, t3, s=1, alpha=1)
		plt.title('Scatter plot of Periodogram Values for All Devices, One Transmission')
		plt.xlabel('Periodogram Value')
		plt.ylabel('Number of Occurances')
		plt.show()
	if line == True:
		plt.plot(x, t3)
		plt.title('Line Graph of Periodogram Values for All Devices, One Transmission')
		plt.xlabel('Periodogram Value')
		plt.ylabel('Number of Occurances')
		plt.show()
	if combinedLine == True:
		for i in range(50):
			#print(arr[(i*3):(i*3) + 3])
			plt.plot([1, 2, 3], arr[(i*3):(i*3) + 3])
		plt.title('Line Graph of Periodogram Values for All Devices, All Transmissions')
		plt.xlabel('Transmission Number')
		plt.ylabel('Periodogram Value')
		plt.show()

# Function to visualize individual packets
# Added breaks, otherwise this would iterate through all packets from all transmissions over all 3 days for a single device
# Returns nothing but displays graphs
def graphPacketSignals():
	deviceNumber = 4
	devPath = "./PacketData/Raw/dev_" + str(deviceNumber) + "_rawPackets.npy"

	packets = np.load(devPath, allow_pickle=True)

	plt.title('Plot of all packets')
	plt.xlabel('time')
	plt.ylabel('signal magnitude')

	for dayData in packets:
		for transmission in dayData:
			for packet in transmission:
				# print(len(packet))

				# Crop the packet since detection is done using the windowed version so there is a little extra noise when using the raw
				croppedPac = packet[100:]
				indx = np.arange(len(croppedPac))
				plt.plot(indx, croppedPac)
				plt.show()
			break
		break

# Function that allows one to get signal data for any packet, any device, any day, from any transmission
# Function takes in the device, day, transmission, and optionally the packet and if it is raw (vs windowed), and returns the packet
def getPacket(device, day, transmission, packet=0, raw=True):
	devPath = "./PacketData/Raw/dev_" + str(device) + "_rawPackets.npy"
	if raw == False:
		devPath = "./PacketData/Windowed/dev_" + str(device) + "_packets.npy"

	deviceInfo = np.load(devPath, allow_pickle=True)

	dayData = deviceInfo[day-1]
	transmissionData = dayData[transmission-1]
	packet = transmissionData[packet]

	if raw == True:
		return packet[100:]
	else:
		return packet

# Function to run seasonal decomposition on a packet
# Function takes in the device to analyze, whether to analyze the raw vs windowed, and if we want to plot the information
# Function returns the resulting information (trend, seasonality, observed, and residual)
def decomposePacket(devNum, raw=True, plot=False):
	devPath = "./PacketData/Raw/dev_" + str(devNum) + "_rawPackets.npy"
	p = 300

	if raw == False:
		devPath = "./PacketData/Windowed/dev_" + str(devNum) + "_packets.npy"
		p = 3

	packets = np.load(devPath, allow_pickle=True)

	firstDay = packets[0]
	firstTrans = firstDay[0]
	firstPacket = firstTrans[0]

	res = seasonal_decompose(firstPacket[100:], period=p)
	if plot == True:
		res.plot()
		plt.show()

	return res