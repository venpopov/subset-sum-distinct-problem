## 22 Nov, 2025

- Earlier today with we discovered that the seed (2,1) beat the Conway-Guy construction. I am running a bunch of different seeds to test for other candidates. 

- massive speed up of the motzkin greedy algorithm [by Gemini Pro 3](https://gemini.google.com/share/0c3abf090162)!

- will systematize results with script later, but some observations:
- seeds that work (produce SSD after a point): 
  - (1,1)
  - (1,2) 
  - (1,3)
  - (1,4) > (1,1) after n18
  - (2,1) > (1,1) after n12
  - (3,2)

(1,7) - (1,9), (4,1), (4,4) big reduction in optimal at high n, but not SSD. Maybe can fix SSD by changing some other value?

(4,2) produces SSD until n=12 and then fails

## 23 Nov, 2025

- I read 
