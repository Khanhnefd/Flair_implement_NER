from flair.models import SequenceTagger
from flair.data import Sentence
from train import PATH_SAVE_MODEL


trained_model = SequenceTagger.load(PATH_SAVE_MODEL+'/best-model.pt')

# sentence to predict
sen = 'Việt Nam vô địch'
sen = Sentence('sen')

trained_model.predict(sen)
print(sen)