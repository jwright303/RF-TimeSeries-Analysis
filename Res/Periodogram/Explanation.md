# Explanation for periodogram analysis of the RF wifi signals

## Important Files
`periodogramRes.txt` - contains results from running the periodogram algorithm on each packet from each transmission for every device. Ißts important to note that the periodogram gives us the dominant cycles of the transmissions, we are keeping the most dominant cycle. The entries show the dominant cycle value and the number of times it has appeared from all of the packets from that single transmission

`PeriodogramResClean.txt` - contains the same as above but with most of the text removed

`dayVals.txt` - contains the most dominant cycle values over all packets and transmissions for the day for every device over all 3 days