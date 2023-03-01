import pandas as pd
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Constants Updates
# Changed LW and NPKT based on observation
NW = 100
LW = 0.01
NPKT = 100

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
	print(len(autoCovar))
	return
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

def getPeriodogram(df, sampleRate=25000000):
	f, Pxx_den = signal.periodogram(df['Val'], sampleRate)
	return f, Pxx_den

# Create the periodogram of the time series
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

def showAutoCorrellation(df):
	res = smt.graphics.plot_acf(df, lags=100)

def showPartialAutoCorrellation(df):
	res = smt.graphics.plot_pacf(df, lags=100)

def plotAllData(df):
	dim = len(df)
	x = [i for i in range(dim)]
	plt.plot(x, df)

def showPlot():
	plt.show()

def getIQData(filename):
	data = np.fromfile(filename, dtype=np.float32)
	sz = len(data[::2])
	i = np.reshape(data[::2], (sz, 1))
	q = np.reshape(data[1::2], (sz, 1))
	#iqArr = np.concatenate((i, q), axis=1)
	
	return i, q

def createMagsArr(i, q):
	mags = np.add(np.square(i), np.square(q))
	df = pd.DataFrame(mags, columns = ['Val'])

	return mags

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

def findPackets(mags):
	packs = []
	rawPacks = []

	sigCount = 0
	sigStart = 0
	sigEnd = 0

	for i in range(len(mags)):
		cur = mags[i]

		if cur >= LW:
			if sigCount == 0:
				sigStart = i

			sigCount += 1

		elif sigCount > 0:
			sigEnd = i-1

			sigLen = sigEnd - sigStart
			if sigLen >= NPKT:
				resCheck = np.array(mags[i:i+22])
				res = resCheck[resCheck[:] > 0.003]
				
				if len(res) < 3:
					#print("sig start, end", sigStart, sigEnd)
					packs.append([sigStart, sigEnd])
					rawPacks.append([sigStart * NW, sigEnd * NW])
			sigCount = 0

	return packs, rawPacks

	