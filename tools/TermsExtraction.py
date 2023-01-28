import spacy
from os import listdir
from os.path import isfile, join
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopWords = set(STOP_WORDS)

def extractTerms(corpus : str = "data/corpus.txt", output : str = "data/terms.txt", minFreq : int = 5):
    """Extract terms from corpus

    Args:
        corpus (str, optional): directory
        output (str, optional): output file name
        minFreq (int, optional): minimum frequency threshold. Defaults to 5.
    """

    AllTerms = dict()
    with open(corpus, "rb") as docText:
        docParsing = nlp(docText.read().decode("utf-8", "ignore"))
        for chunk in docParsing.noun_chunks:
            text = chunk.text.lower()
            words = text.split()
            if words[0] in stopWords:
                np = text.replace(words[0]+ " ", "")
            else:
                np = text
            if np in stopWords:
                continue
            if np in AllTerms.keys():
                AllTerms[np] += 1
            else:
                AllTerms[np] = 1

    with open(output, "w") as f:
        for key in AllTerms.keys():
            if AllTerms[key] >= minFreq:
                f.write(key + "\n")