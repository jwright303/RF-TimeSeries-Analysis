import timeSeriesHelper as tsHelper
import pandas as pd
import numpy as np
from scipy import stats as s

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

def loadIQData(path, name):
	#print("Getting iq data...")
	i, q = tsHelper.getIQData(path + name)
	#print("Creating Mags array...")
	df = tsHelper.createMagsArr(i, q)
	
	return df

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

def parseAllTransmissions(filename):
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

def analyzeAllTransmissions(raw=False):
	for i in range(1, 51):
		print("Device: " + str(i))
		for j in range(3, 6):
			print("Day: " + str(j))
			for k in range(1, 6):
				print("Transmission: " + str(k))
				path = "/Volumes/Jack_SSD/Outdoor/Day_" + str(j) + "/Device_" + str(i) + "/"
				name = "tx_" + str(k) + "_iq.dat"

				df= loadIQData(path, name)
				#print("Creating Windowed Array...")
				res = tsHelper.createWindowedArr(df)
				packs, rawPacks = tsHelper.findPackets(res)
				if raw == True:
					getPeridogramVals(rawPacks, df)
				else:
					getPeridogramVals(packs, res)
				print("")
			print("")
		print("")

def getTransmissionVal(transStr):
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
			key = getTransmissionVal(dictStr)
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

def graphPerDayVals(filename):
	arr = np.fromfile(filename, sep='\n')
	print(len(arr))

#tsHelper.printFunctionality()

#pandasDF = pd.DataFrame(mags, columns = ['Val'])

analyzeAllTransmissions(True)
#parseAllTransmissions("periodogramRes.txt")

#getValPerDay("./Res/periodogramResClean.txt")
#graphPerDayVals("./Res/dayVals.dat")


# df, res = loadIQData(path, name)
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
