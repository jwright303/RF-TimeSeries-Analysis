# Explanation for periodogram analysis of the RF wifi signals

## Important Files
`periodogramRes.txt` - contains results from running the periodogram algorithm on each packet from each transmission for every device. Ißts important to note that the periodogram gives us the dominant cycles of the transmissions, we are keeping the most dominant cycle. The entries show the dominant cycle value and the number of times it has appeared from all of the packets from that single transmission

`PeriodogramResClean.txt` - contains the same as above but with most of the text removed

`dayVals.txt` - contains the most dominant cycle values over all packets and transmissions for the day for every device over all 3 days

## Outcomes
First Transmission</br>
<img width="520" alt="hist-T1" src="https://user-images.githubusercontent.com/41707123/222816495-0d922326-773d-4fe1-ac62-75d7f19ca3d0.png"><img width="520" alt="line-t1" src="https://user-images.githubusercontent.com/41707123/222816525-f3f59545-3ebc-461a-92bc-2f0c097ad6e8.png">

Second Transmission</br>
<img width="520" alt="hist-T2" src="https://user-images.githubusercontent.com/41707123/222816610-06d13420-39ca-478d-8ee4-57fbb40e2cb2.png"><img width="520" alt="line-t2" src="https://user-images.githubusercontent.com/41707123/222816623-ef6c5a5b-1e31-4bc6-8e22-e63e9b0e4891.png">

Third Transmission</br>
<img width="520" alt="hist-T3" src="https://user-images.githubusercontent.com/41707123/222816673-7cd956c5-3cee-4a25-bbe1-422566490ba9.png"><img width="520" alt="line-t3" src="https://user-images.githubusercontent.com/41707123/222816753-941cfd9b-6209-4d59-87bd-22cbce2291f4.png">
