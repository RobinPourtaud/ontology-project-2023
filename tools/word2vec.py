def embedding(corpus, outputModel = "data/model.bin", ouputEmbedding = "data/embedding.txt", minCount = 1):
    """Create word2vec model and embedding file
    Args:
        corpus (str): corpus file
        outputModel (str): output model file
        ouputEmbedding (str): output embedding file
        minCount (int, optional): minimum frequency threshold. Defaults to 1.
    Returns:
        pandas.DataFrame: embedding
    """
    import pandas as pd
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    model = Word2Vec(corpus, window=5, min_count=minCount)
    model.save(outputModel)
    model.wv.save_word2vec_format(ouputEmbedding, binary=False)
    embedding = pd.DataFrame(model.wv.vectors, index = model.wv.index_to_key)
    # add label column
    #embedding['label'] = embedding["label"].apply(lambda x: model.wv.index_to_key[x])
    return embedding