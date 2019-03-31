# ID Agile

ID Agile is a healthcare solution which seeks to reduce patients waiting time (at hospitals queue). Our clients are the health insurance companies and the users are the patients. 

The patient types what he is feeling on our app (e.g. "My chests hurts" or "I have a knee pain when i walk a lot") and then via Natural Language Processing the symptoms are classified into a medical specialty (e.g. "Orthopedist" or "Gastroenterologist"). The app also gathers information such as patients location. The classification will be compared to a database of hospitals and the closest one that has physicians from that specialty will be indicated. 

If the patient decides to go to that hospital, a QR code is generated. The hospital receives an alert on its integrated system and when the patient arrives it simply scans the QR code and he/she will be directed to the correct hospitals wing.

## Implemented
The NLP was implemented making use of SKLearn (Python3). A simple interface was created as well as a SQL database. 

4 ML algorithms were implemented: Decision Tree, Random Forest, Multi-Layer Perceptron and K-Nearest Neighbours. 

Each of them is trained with 60% of the original dataset and tested with the other 40%. The one that performs best (higher accuracy) will be used as the final classifier. 

The dataset used is an adaption from [Medical Speech Transcription and Intent Data](https://www.kaggle.com/paultimothymooney/medical-speech-transcription-and-intent/kernels).

In summary, we applied:

- [x] NLP = Removal of stop words, count vectorizer and tdidf.
- [x] ML = Decision Tree, Random Forest, Multi-Layer Perceptron and K-Nearest Neighbours
- [x] Simplified interface
- [x] Simplified database

![](https://github.com/bzamith/HealthHackathon/blob/master/Pictures/exampleExecution.png)

## Execution
```bash
> pip3 install -r requirements.txt
> python3 interface.py -i overview-of-recordings-especialists.csv 
```

## Files
- ai.py = The artificial intelligence core
- interface.py = Run the interface and calls ai.py
- overview-of-recordings-especialists.csv = Our adapted dataset
- bd.sql = A prototype of a database for our api
- [dashboard.html](https://app.powerbi.com/view?r=eyJrIjoiYTk3MTdiMWMtZjExMS00YjQ5LTgxOWMtYjdmNjM3NzYzNzhkIiwidCI6IjA4MTQ3M2M2LTUwNGEtNDM3Zi04MzhjLWFiOWE2ZjY3MWVmYyIsImMiOjR9)

## Prototype
The screens below are prototypes of our solution:

![](https://github.com/bzamith/HealthHackathon/blob/master/Pictures/prot1.png)
![](https://github.com/bzamith/HealthHackathon/blob/master/Pictures/prot2.png)

## Simulated Dashboard
![](https://github.com/bzamith/HealthHackathon/blob/master/Pictures/dash0.png)
