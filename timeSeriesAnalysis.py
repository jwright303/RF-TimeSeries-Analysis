import timeSeriesHelper as tsHelper
import pandas as pd
import numpy as np
from scipy import stats as s
import matplotlib.pyplot as plt


#Path to the RF data
# path = "/Volumes/Jack_SSD/Outdoor/Day_4/Device_11/"
# name = "tx_3_iq.dat"

def analyzePacket(points, res):
	#print("Packet Points:", points)
	pack = res[(points[0]+1):(points[1])]

	tsHelper.plotAllData(pack)
	tsHelper.showPlot()

	pckDf = pd.DataFrame(pack, columns = ['Val'])
	tsHelper.getStationaryStats(pckDf)

	f, Pxx_den = tsHelper.getPeriodogram(pckDf)
	perdArr = np.array(Pxx_den)
	maxI = np.argmax(perdArr)
	print(maxI)
	#print("max vals", Pxx_den[maxI], f[maxI])
	tsHelper.showPeriodogram(pckDf)
	tsHelper.showPlot()


	tsHelper.showAutoCorrellation(pckDf)
	tsHelper.showPlot()

def plotAllData(data):
	print("Plotting raw data...")
	tsHelper.plotAllData(data)
	tsHelper.showPlot()

def plotWindowedArray(data):
	print("Plotting windowed data...")
	tsHelper.plotAllData(res)
	tsHelper.showPlot()

def analyzePacketPer(points, res):
	#print("Packet Points:", points)
	pack = res[(points[0]+1):(points[1])]

	pckDf = pd.DataFrame(pack, columns = ['Val'])

	f, Pxx_den = tsHelper.getPeriodogram(pckDf)
	perdArr = np.array(Pxx_den)
	maxI = np.argmax(perdArr)

	#print("max vals", f[maxI])
	return f[maxI]

def getPeridogramVals(packs, winArr):
	rSum = 0
	res = {}
	for pacPs in packs:
		v = analyzePacketPer(pacPs, winArr)

		key = int(v)
		if key in res.keys():
			res[key] += 1
		else:
			res[key] = 1
		rSum += v
	print("Per Values", res)
	print("Average Per val:", rSum/float(len(packs)))

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

def getTransmissionPeriodogramVal(transStr):
	dictArr = transStr.split(", ")

	keys = []
	vals = []

	for pair in dictArr:
		vs = pair.split(": ")
		keys.append(int(vs[0]))
		vals.append(int(vs[1]))

	ind = np.argmax(vals)
	#print(keys[ind])
	return keys[ind]

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
	#print("")

	return

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

#tsHelper.printFunctionality()

#pandasDF = pd.DataFrame(mags, columns = ['Val'])

#analyzeAllTransmissions(True)
#parseAllTransmissions("./Res/periodogramRes.txt")
#getValPerDay("./Res/periodogramResClean.txt")

periodogramValsPath = "./Res/Periodogram/dayVals.dat"
#graphPeriodogramDayVals(periodogramValsPath, combinedLine=True)

#Path to the RF data
path = "/Volumes/Jack_SSD/Outdoor/Day_4/Device_11/"
name = "tx_3_iq.dat"
tsHelper.obtainPacketsFromTransmission(raw=True)

#df, res = loadIQData(path, name)
#showAutoCorrellation(df)
# print("Finding packets...")
# packs, rawPacks = tsHelper.findPackets(res)
# print("packs: ", packs)

# print("Plotting packet data...")
# getPeridogramVals(packs)

# print("Plotting graphs...")
# plotAllData(df)
# plotWindowedArray(res)

# model = ARIMA(df)
# model_fit = model.fit()
#print(model_fit.summary())

# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# plt.show()

# model_fit.plot_diagnostics()
# plt.show()
