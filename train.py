from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from data_embedding import get_corpus, get_embedding

# config
EPOCHS = 70
LR = 0.1
BATCH_SIZE = 32
PATH_SAVE_MODEL = '/PATH/SAVE/NER/MODEL'


embedding_VN = get_embedding()
corpus, label_type, label_dict = get_corpus()

# train model
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embedding_VN,
                        tag_type=label_type,
                        tag_dictionary=label_dict,
                        use_crf=True)

trainer = ModelTrainer(tagger, corpus)
trainer.train(PATH_SAVE_MODEL,
              learning_rate=LR,
              mini_batch_size=BATCH_SIZE,
              max_epochs=EPOCHS,
              checkpoint=True)





                        
