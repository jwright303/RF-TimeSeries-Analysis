import pandas as pd
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

# This is a file with a lot of the helper functions for Time Series Analysis of RF data
# To see all of the functionality the printFunctionality Function can be called

# Constants Updates
# Changed LW and NPKT based on observation
NW = 100		# Number of signals per window
LW = 0.01		# Threshold for the window magnitude
NPKT = 100		# Number of windows for it to be considered a packet

def printFunctionality():
	print("Welcome these are the modules I created for time series analysis of RF data")
	print("Functions Available:")
	print("	*	Display the AutoCovariance")
	print("	*	Display the Periodogram")
	print("	*	Display the Signal Histogram")
	print("	*	Display the AutoCorrellation")
	print("	*	Display the partial AutoCorrellation")
	print("	*	Display all magnitude data")
	print("	*	Windowize data")


def calculateAutoCovariance(df):
	autoCovar = smt.stattools.acovf(df["Val"])
	indx = [i for i in range(len(df['Val']))]
	plt.plot(indx, autoCovar)
	#plt.scatter(indx, autoCovar, s=1, alpha=1)
	#plt.hist(autoCovar, bins=128, color='#fcba03')
	plt.title('Plot of autoCovar')
	plt.xlabel('autoCovar')
	plt.ylabel('Frequency')

# ADF stat tells us about the differencing needed to make it stationary
# P-value gives us an idea on the confidence in this, we want it less than 0.05
def getStationaryStats(df):
	result = adfuller(df)

	print(df.describe())
	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])

# Function returns the periodogram of a time series given the time series as a data frame and the sample rate used to collect the points
def getPeriodogram(df, sampleRate=25000000):
	f, Pxx_den = signal.periodogram(df['Val'], sampleRate)
	return f, Pxx_den

# Create and show periodogram of the time series
def showPeriodogram(df, sampleRate=25000000):
	f, Pxx_den = signal.periodogram(df['Val'], sampleRate)
	plt.semilogy(f, Pxx_den)
	plt.xlabel('frequency [Hz]')
	plt.ylabel('PSD')

# Histogram of the magnitudes of the signal
def showSignalHistogram(df):
	plt.hist(df['Val'], bins=128, color='#fcba03')
	plt.title('Histogram of Magnitudes')
	plt.xlabel('Signal Magnitudes')
	plt.ylabel('Frequency')

# Plots the autocorrelation of a time series when given the time series as a data frame
def showAutoCorrellation(df):
	res = smt.graphics.plot_acf(df, lags=100)

# Plots the partial autocorrelation of a time series when given the time series as a data frame
def showPartialAutoCorrellation(df):
	res = smt.graphics.plot_pacf(df, lags=100)

# Plots the entire time series data
def plotAllData(df):
	dim = len(df)
	x = [i for i in range(dim)]
	plt.plot(x, df)

# Will show whatever plots are staged to be displayed
def showPlot():
	plt.show()

# Reads in the Iq data from a given file name / path, assumes that its I first then Q
# Returns the I and Q data as seperate arrays
def getIQData(filename):
	data = np.fromfile(filename, dtype=np.float32)
	sz = len(data[::2])
	i = np.reshape(data[::2], (sz, 1))
	q = np.reshape(data[1::2], (sz, 1))
	#iqArr = np.concatenate((i, q), axis=1)
	
	return i, q

# Creates an array of signal magnitudes
# Simply squares both values and adds them together to get the magnitude and returns a dataframe with the results
def createMagsArr(i, q):
	mags = np.add(np.square(i), np.square(q))
	df = pd.DataFrame(mags, columns = ['Val'])

	return mags

# Creates a windowed array from the magnitude array
# Essentially downsaples the origional arraay by summing each window and using that as the new value
def createWindowedArr(mags):
	smpls = len(mags)
	numWindows = (-1 * (-smpls // NW))
	res = []

	startIdx = 0
	endIdx = NW

	for i in range(0, numWindows):
		cur = mags[startIdx : endIdx]

		mg = np.sum(cur)
		res.append(mg)

		startIdx = endIdx
		# if endIdx + NW > smpls:
		# 	endIdx += (smpls - startIdx)
		# else:
		endIdx += NW

	return res

# Function to find packets from a magnitude array - somewhat experimental
# Works by checking the threshold of the magnitude array and if it meets then we count it as a signal
# We also have a threshold for the number of signals
#	If there are NPKT enough signals then I check to see if there is a quite time after from the reserved channel from the wifi
# Returns an array specifying the start and end indecies of all the registered packets of both the windowed array and the raw magnitude array
def findPackets(mags, windowedMags):
	packs = []
	rawPacks = []

	sigCount = 0
	sigStart = 0
	sigEnd = 0

	for i in range(len(windowedMags)):
		cur = windowedMags[i]

		if cur >= LW:
			if sigCount == 0:
				sigStart = i

			sigCount += 1

		elif sigCount > 0:
			sigEnd = i-1

			sigLen = sigEnd - sigStart
			if sigLen >= NPKT:
				resCheck = np.array(windowedMags[i:i+22])
				res = resCheck[resCheck[:] > 0.003]
				
				if len(res) < 3:
					#print("sig start, end", sigStart, sigEnd)
					packs.append(windowedMags[sigStart+1:sigEnd])
					rawPacks.append(mags[(sigStart * NW) + 1:sigEnd * NW])
			sigCount = 0

	return packs, rawPacks

# Function to read all packet data from a given path
# There is also an option to read in the packet data from the raw signals compared to the windowed signals
# Returns an array filled with transmissions from all 50 devices as well as an optional array filled with raw signals
def readAllPacketData(path, raw=False, rawPath=""):
	rawDevices = []
	windowedDevices = []
	for i in range(1, 51):
		windowedDevices.append(np.load(path + "dev_" + str(i) + "_packets.npy"))
		if raw == True:
			rawDevices.append(np.load(rawPath + "dev_" + str(i) + "_rawPackets.npy"))

	return windowedDevices, rawDevices

# Function to read packet data for a range of devices
# Takes in the path of the desired packet information as well as the starting and ending device requested
# Returns an array of packet data
def readSelectedPacketData(path, start, end):
	devices = []
	for i in range(start, end+1):
		devices.append(np.load(path + "dev_" + str(i) + "_packets.npy"))
	return devices

# Function to save packet info for raw and windowed signals
# Takes in an array of windowed signal data and raw signal data for every transmission of a deivce, as well as the device number
# Saves info to a predetermined path and returns nothing 
def savePacketInfo(rawPacks, packs, deviceNum):
	rawPath = "PacketData/Raw/"
	windowedPath = "PacketData/Windowed/"

	with open(rawPath + 'dev_' + str(deviceNum) + '_rawPackets.npy', 'wb') as f:
		np.save(f, np.array(rawPacks, dtype=object))
	with open(windowedPath + 'dev_' + str(deviceNum) + '_packets.npy', 'wb') as f:
		np.save(f, np.array(packs, dtype=object))

# Function to obtain signal data from every detected packet for every device over all 3 days
# Takes in the option to also obtain raw signal data for the packets as well (ropughly 100 times more points per signal)
# Returns nothing, but has the packet data saved
def obtainPacketsFromTransmission(raw=False):

	for n in range(1, 51):
		print("Device: " + str(n))
		rawDayInfo = []
		dayInfo = []
		for j in range(3, 6):
			rawTransmissionInfo = []
			transmissionInfo = []
			print("Day: " + str(j))
			for k in range(1, 6):
				print("Transmission: " + str(k))
				path = "/Volumes/Jack_SSD/Outdoor/Day_" + str(j) + "/Device_" + str(n) + "/"
				name = "tx_" + str(k) + "_iq.dat"

				i, q = getIQData(path + name)
				#print("Creating Mags array...")
				df = createMagsArr(i, q)

				#print("Creating Windowed Array...")
				res = createWindowedArr(df)

				packs, rawPacks = findPackets(df, res)

				rawTransmissionInfo.append(rawPacks)
				transmissionInfo.append(packs)

			rawDayInfo.append(rawTransmissionInfo)
			dayInfo.append(transmissionInfo)

		savePacketInfo(rawDayInfo, dayInfo, n)

	