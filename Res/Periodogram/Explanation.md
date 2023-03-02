# Explanation for periodogram analysis of the RF wifi signals

## Important Files
`periodogramRes.txt` - contains results from running the periodogram algorithm on each packet from each transmission for every device. Ißts important to note that the periodogram gives us the dominant cycles of the transmissions, we are keeping the most dominant cycle. The entries show the dominant cycle value and the number of times it has appeared from all of the packets from that single transmission

`PeriodogramResClean.txt` - contains the same as above but with most of the text removed

`dayVals.txt` - contains the most dominant cycle values over all packets and transmissions for the day for every device over all 3 days

## Outcomes
First Transmission</br>
<img width="558" alt="histogram first transmission" src="https://user-images.githubusercontent.com/41707123/222556154-edd080c6-ad6d-4db7-a6b1-6e0cb581953a.png">

<img width="551" alt="plot first transmission" src="https://user-images.githubusercontent.com/41707123/222556212-bf3da8b8-9515-44f3-9f97-1dc3a089fd34.png">

Second Transmission</br>
<img width="558" alt="histogram second transmission" src="https://user-images.githubusercontent.com/41707123/222556252-7b95701f-9dad-40ee-902d-84e2fc692f2e.png">

<img width="552" alt="plot second transmission" src="https://user-images.githubusercontent.com/41707123/222556275-a61cd6bd-d231-485f-a6a7-f6fe9ea12015.png">

Third Transmission</br>
<img width="558" alt="histogram third transmission" src="https://user-images.githubusercontent.com/41707123/222556312-f936ee8c-7202-4928-9657-b738e5cb40c4.png">

<img width="556" alt="plot third transmission" src="https://user-images.githubusercontent.com/41707123/222556324-1741b82b-f2c0-4eb5-8ac1-142037ec178f.png">
