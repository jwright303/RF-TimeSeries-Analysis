import pandas as pd
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


path = "/Volumes/Jack_SSD/Outdoor/Day_4/Device_3/"
name = "tx_1_iq.dat"

# Calculate the auto-covariance values of the time series
def calculateAutoCovariance(df):
	autoCovar = smt.stattools.acovf(df["Val"])
	print(len(autoCovar))
	return
	indx = [i for i in range(len(df['Val']))]
	plt.plot(indx, autoCovar)
	#plt.scatter(indx, autoCovar, s=1, alpha=1)
	#plt.hist(autoCovar, bins=128, color='#fcba03')
	plt.title('Histogram of autoCovar')
	plt.xlabel('autoCovar')
	plt.ylabel('Frequency')
	plt.show()

# Create the periodogram of the time series
def showPeriodogram(df):
	sampleRate = 25000000
	f, Pxx_den = signal.periodogram(df['Val'], sampleRate)
	plt.semilogy(f, Pxx_den)
	plt.xlabel('frequency [Hz]')
	plt.ylabel('PSD')
	plt.show()

# Histogram of the magnitudes of the signal
def showSignalHistogram(df):
	plt.hist(df['Val'], bins=128, color='#fcba03')
	plt.title('Histogram of Magnitudes')
	plt.xlabel('Signal Magnitudes')
	plt.ylabel('Frequency')
	plt.show()

def showAutoCorrellation(df):
	res = smt.graphics.plot_acf(df, lags=100)
	plt.show()

#df = pd.read_table(path + name, sep=)

data = np.fromfile(path + name, dtype=np.float32)
#print(data)

sz = len(data[::2])
i = np.reshape(data[::2], (sz, 1))
q = np.reshape(data[1::2], (sz, 1))
iqArr = np.concatenate((i, q), axis=1)

mags = np.add(np.square(i), np.square(q))

df = pd.DataFrame(mags[:5000], columns = ['Val'])
print(df.describe())

showPeriodogram(df)
#calculateAutoCovariance(df)
#showAutoCorrellation(df)

