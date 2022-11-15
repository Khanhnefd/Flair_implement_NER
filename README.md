# Flair_implement_NER
Using Flair for Named Entity Recognition in NLP

1. Dataset
    - Small NER dataset (Vietnamese)  => download: [here](https://drive.google.com/drive/folders/175Bo9JFwBorAr_2jlSlvoqJMgDj-LW9U?usp=sharing)
    - Contains 65 tags: O, S-PERSON, B-PERSON, E-PERSON, I-PERSON, S-ORGANIZATION, ...
    - 3 files: train.csv, test.csv and dev.csv

2. Embeddings:
    - Use glove.840B.300d  => [here](https://nlp.stanford.edu/projects/glove/)

3. Requirements:
    - Flair: pip install flair => [link docs](https://github.com/flairNLP/flair)
    - Python >= 3.7

4. Training:
    - Change GLOVE_PATH, DATA_PATH to your proper file location and other constant to whatever name you want
    - Run train.py

5. Result:
    - After training model with this dataset for 70 epochs:
       
   | Metric        | Result           | 
   | ------------- |:-------------:| 
   | F-score (micro)| 0.547 |
   | F-score (macro)| 0.2854  |  
   | Accuracy | 0.4036 |
