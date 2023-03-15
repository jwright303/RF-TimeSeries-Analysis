import pytest
import sys
import importlib

sys.path.append('../RF-TimeSeries-Analysis')
tsA = importlib.import_module("timeSeriesAnalysis")
tsH = importlib.import_module("timeSeriesHelper")

#Get first device, first day, first transmission, and first packet
def test_read_packet():
  samplePacket = tsA.getPacket(1, 1, 1, raw=False)
  assert samplePacket != None

# res = tsA.decomposePacket(1, raw=False)
# assert res.all() != None
def test_analyze_packet():
  samplePacket = tsA.getPacket(1, 1, 1, raw=False)
  res = tsA.analyzePacketPer(samplePacket)
  assert res != None
