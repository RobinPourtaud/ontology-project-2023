

def corpusParsing(min_freq = 2, corpus = "data/corpus.txt", output_file = "data/processedCorpus.txt", freqTerms_output_file = "data/freqTerms.txt", isDep = True):
    """ Create the processed corpus file and the frequent terms file
    Args:
        min_freq (int, optional): minimum frequent threshold for creating the file of frequent terms. Defaults to 2.
        corpus (str, optional): directory for corpus documents. Defaults to "data/corpus.txt".
        output_file (str, optional): name of the output file (the processed corpus). Defaults to "data/processedCorpus.txt".
        freqTerms_output_file (str, optional): name of the output file (the frequent terms). Defaults to "data/freqTerms.txt".
        isDep (bool, optional): True for depenceny parsing; False for shallow parsing. Defaults to True.
    Returns:
        None
    """
    import spacy
    nlp = spacy.load('en_core_web_sm')
    from spacy.lang.en.stop_words import STOP_WORDS
    stopWords = set(STOP_WORDS)
    lemmas = dict()
    doc = corpus
    with open(output_file, "w") as f:
        f.write("<text>" + "\n") # new document
        with open(doc, "rb")as docText:
            for line in docText:
                f.write("<s>" + "\n") #new sentence
                sent = line.strip().decode("utf-8", "ignore")
                if isDep:
                    parsedSent = nlp(sent) #dependency parsing
                else:
                    parsedSent = nlp(sent, disable=['parser']) #shallow parsing
                index = 0
                for token in parsedSent:
                    if isDep:
                        w = token.text + "\t" + token.lemma_ + "\t" + token.pos_ + "\t" + str(
                            index) + "\t" + token.head.text + "\t" + token.dep_ + "\n"
                    else:
                        w = token.text + "\t" + token.lemma_  + "\t" + token.pos_ + "\t" + str(
                            index) + "\tparent\tdep\n"
                    f.write(w) #sentence word
                    index += 1
                    lemma = token.lemma_
                    if lemma in lemmas.keys():
                        lemmas[lemma] = lemmas[lemma] + 1
                    else:
                        lemmas[lemma] = 1
                f.write("</s>" + "\n")
        f.write("</text>" + "\n")
        
    #write the frequent lemmas into frequnet file
    with open(freqTerms_output_file, "w") as freq_file:
        for key in lemmas.keys()  :
            if lemmas[key] >= min_freq:
                if key in stopWords:
                    continue
                else:  
                    freq_file.write(key + "\n")