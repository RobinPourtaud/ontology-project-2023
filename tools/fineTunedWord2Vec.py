def embedding(corpus, outputModel = "data/model.bin", ouputEmbedding = "data/embedding.txt"):
    """Create word2vec model and embedding file
    Args:
        corpus (str): corpus file
        outputModel (str): output model file
        ouputEmbedding (str): output embedding file
    Returns:
        pandas.DataFrame: embedding
    """
    import pandas as pd
    from gensim.models import Word2Vec
    