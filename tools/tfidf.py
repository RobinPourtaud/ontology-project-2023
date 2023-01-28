def get_sentences(corpus_file):
    """
    Returns all the (content) sentences in a corpus file
    :param corpus_file: the corpus file
    :return: the next sentence (yield)
    """

    # Read all the sentences in the file
    with open(corpus_file, 'r', errors='ignore') as f_in:

        s = []

        for line in f_in:
            line = line

            # Ignore start and end of doc
            if '<text' in line or '</text' in line or '<s>' in line:
                continue
            # End of sentence
            elif '</s>' in line:
                yield s
                s = []
            else:
                try:
                    word, lemma, pos, index, parent, dep = line.split()
                    s.append((word, lemma, pos, int(index), parent, dep))
                # One of the items is a space - ignore this token
                except Exception as e:
                    print (str(e))
                    continue

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim import corpora, similarities, models
import jieba
import spacy
from os import listdir
from os.path import isfile, join
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopWords = set(STOP_WORDS)
dir = r".\Corpus" #directory for corpus documents
output_dir = r".\OutputDir" #result files output directory
corpus=[]
alldocs = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
for doc in alldocs:
    docText = open(doc, "r", errors='ignore').read()
    docParsing = nlp(docText)
    corpus.append(str(docParsing))

def removeArticles(text):
    #remove stop words from the begining of a NP
    words = text.split()
    if words[0] in stopWords:
        return text.replace(words[0]+ " ", "")
    return text

def getSentence(strip_sentence):
    """
            Returns sentence (space seperated tokens)
            :param strip_sentence: the list of tokens with other information for each token
            :return: the sentence as string
    """
    sent = ""
    for i, (t_word, t_lemma, t_pos, t_index, t_parent, t_dep) in enumerate(strip_sentence):
        sent += t_word.lower() + " "
        print(sent)
    return sent

processed_corpus_dir = r".\OutputDir"
corpus_files = sorted([processed_corpus_dir + '/' + file for file in listdir(processed_corpus_dir) if str(file).__contains__("processed")])
for file_num, corpus_file in enumerate(corpus_files):
    sen=get_sentences(corpus_file)
    break

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

word = vectorizer.get_feature_names()

transformer = TfidfTransformer()
# Calculate the TF-IDF value
tfidf = transformer.fit_transform(X)
# Calculate the weight of each word
weight = tfidf.toarray()