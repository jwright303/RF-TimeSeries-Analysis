import timeSeriesAnalysis as tsA
import timeSeriesHelper as tsH
import pytest

#Get first device, first day, first transmission, and first packet
samplePacket = tsA.getPacket(1, 1, 1)
assert samplePacket.all() != None

res = tsA.analyzePacketPer(samplePacket)
assert res != None