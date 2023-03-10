from __future__ import print_function
from scapy.all import ( Dot11,
                        Dot11Beacon,
                        Dot11Elt,
                        RadioTap,
                        sendp,
                        hexdump)
SSID = 'Test SSID'
iface = 'wlp2s0'
sender = 'ac:cb:12:ad:58:27'

dot11 = Dot11(type=0, subtype=8, addr1='ff:ff:ff:ff:ff:ff', addr2=sender, addr3=sender)
beacon = Dot11Beacon()
essid = Dot11Elt(ID='SSID',info=SSID, len=len(SSID))

frame = RadioTap()/dot11/beacon/essid
for b in str(frame):
  print ("char: %s ord/value: %d hex: %x", (b, ord(b), ord(b)))
hexdump(frame)
