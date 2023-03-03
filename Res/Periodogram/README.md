# Explanation for periodogram analysis of the RF wifi signals

## Important Files
`periodogramRes.txt` - contains results from running the periodogram algorithm on each packet from each transmission for every device. Ißts important to note that the periodogram gives us the dominant cycles of the transmissions, we are keeping the most dominant cycle. The entries show the dominant cycle value and the number of times it has appeared from all of the packets from that single transmission

`PeriodogramResClean.txt` - contains the same as above but with most of the text removed

`dayVals.txt` - contains the most dominant cycle values over all packets and transmissions for the day for every device over all 3 days

## Outcomes
The first thing to look at is the histogram and the line graph of the periodogram values. The histogram of these values gives us an idea on how unique the these values are. The line graph gives us an idea on how these values change from device to device. A very important thing to note is the way in which these transmissions were collected. The method for collecting the transmission was to have 10 devices transmit sequentially (device 1 then 2 and so on) which was repeated 3 times before moving on to the next batch of 10 devices. The fingerprint of devices is very dependent on the time when they were transmitted which is part of the reason we can see the periodogram values change from batch to batch.

First Transmission</br>
<img width="500" alt="hist-T1" src="https://user-images.githubusercontent.com/41707123/222816495-0d922326-773d-4fe1-ac62-75d7f19ca3d0.png"><img width="500" alt="line-t1" src="https://user-images.githubusercontent.com/41707123/222816525-f3f59545-3ebc-461a-92bc-2f0c097ad6e8.png"></br>
The first transmission shows that the periodogram values are very similar and the change from device to device is generally very small.

Second Transmission</br>
<img width="500" alt="hist-T2" src="https://user-images.githubusercontent.com/41707123/222816610-06d13420-39ca-478d-8ee4-57fbb40e2cb2.png"><img width="500" alt="line-t2" src="https://user-images.githubusercontent.com/41707123/222816623-ef6c5a5b-1e31-4bc6-8e22-e63e9b0e4891.png"></br>
The second transmisison shows that the periodogram values are also quite similar to each other from the histogram. One thing to note is that there is a drastic change in value range from the first 20 devices to the next 30 devices. It's also worth noting that the last 30 devices have much more random values.

Third Transmission</br>
<img width="500" alt="hist-T3" src="https://user-images.githubusercontent.com/41707123/222816673-7cd956c5-3cee-4a25-bbe1-422566490ba9.png"><img width="500" alt="line-t3" src="https://user-images.githubusercontent.com/41707123/222816753-941cfd9b-6209-4d59-87bd-22cbce2291f4.png">
The third transmission is one of the most interesting. The values are the most unique with still a good chunk of repeating values.

Periodogram values for all devices over the three transmissions
<img width="706" alt="Screenshot 2023-03-03 at 11 45 52 AM" src="https://user-images.githubusercontent.com/41707123/222822102-34f93c2a-6c56-44fc-a492-6207d9dc509e.png">
The graph of values changing for each device over the 3 transmission is another one of the most interesting displays. The values all start in a similar range, diverge during the second transmission, and converge again to a more broad range for the third transmission.
