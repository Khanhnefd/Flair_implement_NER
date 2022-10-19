from flair.data import Corpus
from flair.datasets import ColumnCorpus
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from flair.embeddings import WordEmbeddings


GLOVE_PATH = 'PATH/TO/YOUR/GLOVE/FILE'
OUTPUT_FILE = '/OUTPUT/FILE'
FILE_KEY_VECTOR = '/PATH/SAVE/KEY/VECTOR'
DATA_PATH = 'PATH/TO/YOUR/DATA/FOLDER'

def get_embedding():
    # process glove embedding
    input_file = GLOVE_PATH
    output_file = OUTPUT_FILE
    glove2word2vec(input_file, output_file)
    key_vector = KeyedVectors.load_word2vec_format(output_file, binary=False)

    # save to file
    key_vector.save(FILE_KEY_VECTOR)

    # get WordEmbedding for Flair
    embedding_VN = WordEmbeddings(FILE_KEY_VECTOR)

    return embedding_VN


def get_corpus():
    data_path = DATA_PATH

    # column format indicating which columns hold the text and label(s)
    column_name_map = {3: "ner", 0: "text"}

    corpus: Corpus = ColumnCorpus(data_path, column_name_map,
                                  test_file='test.csv',
                                  train_file='train.csv',
                                  dev_file='dev.csv'
                                  )

    # label of corpus
    label_type = 'ner'
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    return corpus, label_type, label_dict
