# ID Agile


## Execution
At this same dir:
```bash
> pip3 install -r requirements.txt
> python3 interface.py -i overview-of-recordings-especialists.csv 
```

## Implementation
* Python 3 (_não compatível com versões anteriores_)
* Pandas Dataframe
* SKLearn

## Files
- ai.py = The artificial intelligence core
- interface.py = Run the interface and call ai 
- overview-of-recordings-especialists.csv = Our adapted dataset
- overview-of-recordings.csv = Original dataset ([source](https://www.kaggle.com/paultimothymooney/medical-speech-transcription-and-intent/kernels))
- bd.sql = A prototype of a database for our api

### Main implementations
- [x] NLP 
- [x] Decision Tree, Random Forest, MLP and KNN
- [x] Simple interface
