class Embedding: 
    def __init__(self):
        self.y = {}
        self.yPred = {}
        self.labelClass = {}
        self.embedding = {}
        self.mainCorpus = None
        self.key = {}
        self.model = {}
        self.freq = {}

    def loadCorpus(self, corpus): 
        """Load corpus in the format of a list (corpus) of list (sentences) of string (words)

        Args:
            corpus (list): corpus
        """
        assert type(corpus) == list, "Corpus must be a list"
        assert type(corpus[0]) == list, "Corpus must be a list of list"
        assert type(corpus[0][0]) == str, "Corpus must be a list of list of string"
        self.mainCorpus = corpus

    def loadLabel(self, file = "data/NP.csv", online = True): 
        """Load labelabulary from a local file or from a URL
        Args:
            label (str): labelabulary file (ignored if online = True)
            online (bool, optional): Load data from a local file or from a URL. Defaults to True.
        """
        import pandas as pd
        column = ["noun_phrase","frequency","Core concept"]
        if online: 
            try : 
                d = pd.read_csv('https://docs.google.com/spreadsheets/d/16OlZEn2__3ALyECxYS00vlkECWGORTsfKs3ygWjaue8/export?gid=0&format=csv')[column].dropna()
            except Exception as e:
                print("Error: ", e)
                print("Loading data from local file")
        d = pd.read_csv(file)[column].dropna()
        
        self.y = {np : Concept for np, Concept in zip(d["noun_phrase"], d["Core concept"])}
        self.freq = {np : freq for np, freq in zip(d["noun_phrase"], d["frequency"])}
        print("Vocabulary loaded : self.label")
        print("Labels loaded : self.y")

    def filterEmbedding(self, embeddingType = "word2vec", minFreq = None):
        """Filter embedding with vocabulary

        Args:
            embeddingType (str): embeddingType of embedding
            minFreq (int): minimum frequency of a word to be included in the embedding
        """
        if minFreq != None: 
            assert type(minFreq) == int, "freq must be an integer"
            assert minFreq > 0, "freq must be greater than 0"
            self.y = {np : Concept for np, Concept in self.y.items() if self.freq[np] >= minFreq}
        yPresents = self.embedding[embeddingType].index.intersection(self.y)
        if len(yPresents) != len(self.y):
            print("Warning: {} labels are not in the embedding".format(len(self.y) - len(yPresents)))
        self.y = {np : Concept for np, Concept in self.y.items() if np in yPresents}
        self.embedding[embeddingType] = self.embedding[embeddingType].loc[yPresents]
    
    def loadKey(self, key : str, keyType = "OpenAI"):
        """Add key for accessing API

        Args:
            key (str): key
            keyType (str, optional): keyType of API. Defaults to "OpenAI".

        """
        print("If you want to use OpenAI API, please visit https://beta.openai.com/docs/api-reference/authentication to get your key.")
        self.key[keyType] = key
        print("Key loaded : self.key[\"{}\"]".format(keyType))

    def computeWord2Vec(self, outputModel = "data/model.bin", ouputEmbedding = "data/W2Vembedding.txt"):
        """Create word2vec model and embedding file
        Args:
            corpus (str): corpus file
            outputModel (str): output model file
            ouputEmbedding (str): output embedding file
        """
        import pandas as pd
        from gensim.models import Word2Vec
        assert self.mainCorpus != None, "Please load corpus first with loadCorpus"
        model = Word2Vec(self.mainCorpus, window=5, min_count=1)
        model.train(self.mainCorpus, total_examples=len(self.mainCorpus), epochs=100)
        model.save(outputModel)
        model.wv.save_word2vec_format(ouputEmbedding, binary=False)
        embedding = pd.DataFrame(model.wv.vectors, index = model.wv.index_to_key)

        self.embedding["word2vec"] = embedding
        print("Word2Vec embedding computed : self.embedding[\"word2vec\"]")

    def computeWord2Vec2(self, outputModel = "data/model.bin", ouputEmbedding = "data/W2V2embedding.txt"):
        """Create word2vec model and embedding file
        Args:
            corpus (str): corpus file
            outputModel (str): output model file
            ouputEmbedding (str): output embedding file
        """
        import pandas as pd
        from gensim.models import Word2Vec
        assert self.mainCorpus != None, "Please load corpus first with loadCorpus"
        c = [list(self.y.keys()) for i in range(100)]
        model = Word2Vec(c, window=5, min_count=1)
        model.train(c, total_examples=len(c), epochs=100)
        model.save(outputModel)
        model.wv.save_word2vec_format(ouputEmbedding, binary=False)
        embedding = pd.DataFrame(model.wv.vectors, index = model.wv.index_to_key)

        self.embedding["word2vec2"] = embedding
        print("Word2Vec embedding computed : self.embedding[\"word2vec\"]")

    def computeWord2VecGNews(self, gNewsBin = "data/gNews.bin", outputModel = "model/gNewsDoc2Vec.bin", ouputEmbedding = "data/embeddingGNews.txt"):
        """Create word2vec model and embedding file on gNews
        Args:
            outputModel (str): output model file
            ouputEmbedding (str): output embedding file
 
        """
        import pandas as pd
        from gensim.models import Word2Vec
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load_word2vec_format(gNewsBin, binary=True)
        embedding = pd.DataFrame(kv.vectors, index = kv.index_to_key)

        self.embedding["word2vecGNews"] = embedding
        print("Word2Vec embedding computed : self.embedding[\"word2vecGNews\"]")

    def computeWord2VecEnWiki(self, enWikiTxt = "data/enwiki_20180420_500d.txt", outputModel = "model/enWikiDoc2Vec.bin", ouputEmbedding = "data/embeddingEnWiki.txt"):
        """Create word2vec model and embedding file on enWiki
        Args:
            outputModel (str): output model file
            ouputEmbedding (str): output embedding file
 
        """
        import pandas as pd
        from gensim.models import Word2Vec
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load_word2vec_format(enWikiTxt, binary=False)
        embedding = pd.DataFrame(kv.vectors, index = kv.index_to_key)

        self.embedding["word2vecEnWiki"] = embedding
        print("Word2Vec embedding computed : self.embedding[\"word2vecEnWiki\"]")

    def computeGPT(self,  outputEmbedding = "data/GPTembedding.txt", onlyWords = None):
        """Create GPT model and embedding file
        
        Args:
            outputEmbedding (str): output embedding file
            onlyWords (list, optional): list of words to compute embedding. Defaults to None (all)

        Returns:
            pandas.DataFrame: embedding for each word in labelabulary
        """
        import pandas as pd
        import openai
        assert "OpenAI" in self.key, "OpenAI key is not set, please use loadKey(key, embeddingType = 'OpenAI'): to set it"
        openai.api_key = self.key["OpenAI"]
        import time
        # create embedding
        embedding = {}
        if onlyWords == None:
            w = list(self.y.keys())
        else:
            w = onlyWords
        for word in w:
            print(word)
            try :
                response = openai.Embedding.create(input = word, model = "text-embedding-ada-002")
                embedding[word] = " ".join([str(x) for x in response["data"][0]["embedding"]])
                with open(outputEmbedding, "a") as f:
                    f.write(word + " " + embedding[word] + "\n")
            except:
                print("API pause of 61s, too much requests")
                time.sleep(61)
                # yes, not clean
                response = openai.Embedding.create(input = word, model = "text-embedding-ada-002")
                embedding[word] = " ".join([str(x) for x in response["data"][0]["embedding"]])
                with open(outputEmbedding, "a") as f:
                    f.write(word + " " + embedding[word] + "\n")
                
        d = pd.DataFrame.from_dict(embedding, orient = "index")
        self.embedding["GPT"] = d[0].str.split(" ", expand = True)
        print("GPT embedding computed : self.embedding[\"GPT\"]")


    def plotEmbedding(self, embeddingType = "word2vec", dim = 2, reduction = "ACP", y = "y", showLinks = False): 
        """Plot embedding
        
        Args:
            embeddingType (str, optional): type of embedding. Defaults to "word2vec".
            dim (int, optional): dimension of embedding. Defaults to 2.
            reduction (str, optional): reduction method. Defaults to "ACP".

        Returns:
            matplotlib.figure.Figure: figure
        """
        assert embeddingType in self.embedding, "Embedding embeddingType is not computed, please use computeWord2Vec() or computeGPT() to compute it"
        assert reduction in ["ACP", "TSNE"], "Reduction method is not supported"
        assert len(self.embedding[embeddingType]) == len(self.y), "Embedding size is : " + str(len(self.embedding[embeddingType])) + " and y size is : " + str(len(self.y)) + ", use filterEmbedding() to filter embedding"
        assert dim in [2, 3], "Dimension is not supported"
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import plotly.graph_objects as go
        import plotly.express as px
        # get embedding
        embedding = self.embedding[embeddingType]

        if y == "y":
            yD = self.y
        else:
            assert y in self.yPred, "y is not in yPred, please use SVM() or HDBSCAN() to compute it"
            yD = self.yPred[y]

        # reduce dimension
        if reduction == "ACP":
            pca = PCA(n_components=dim)
            embedding = pca.fit_transform(embedding)
        elif reduction == "TSNE":
            tsne = TSNE(n_components=dim, perplexity=min(max(len(y)-1, 1), 30))
            embedding = tsne.fit_transform(embedding)
        # plot
        if dim == 2:
            fig = px.scatter(x=embedding[:,0], y=embedding[:,1], color = yD, hover_name=self.y.keys())
        elif dim == 3:
            fig = px.scatter_3d(x=embedding[:,0], y=embedding[:,1], z=embedding[:,2], color = yD, hover_name=self.y.keys())

        if y == "HDBSCAN" and showLinks:
            links = self.model["HDBSCAN"]._min_spanning_tree
            idIn = [int(i) for i in links[:,0]]
            idOut = [int(i) for i in links[:,1]]
            idWeight = [i for i in links[:,2]]
            for i in range(len(links)):
                f = go.Scatter3d(x=[embedding[idIn[i],0], embedding[idOut[i],0]], y=[embedding[idIn[i],1], embedding[idOut[i],1]], z=[embedding[idIn[i],2], embedding[idOut[i],2]], mode='lines', line=dict(color=idWeight[i], width=3), showlegend=False)
                # delete legend
                f.update(showlegend=False)
                fig.add_trace(
                    f
                )

                
        fig.update_layout(
            title = "Embedding " + embeddingType + " with " + reduction + " reduction",
            legend_title="Cluster",
            xaxis_title="x",
            yaxis_title="y",

        )
        return fig


    def computeSVC(self, embeddingType = "word2vec", kernel = "linear", C = 1.0, gamma = "scale", testSize = 0.2):
        """Compute SVC model
        
        Args:
            embeddingType (str, optional): type of embedding. Defaults to "word2vec".
            kernel (str, optional): kernel. Defaults to "linear".
            C (float, optional): C parameter. Defaults to 1.0.
            gamma (str, optional): gamma parameter. Defaults to "scale".
            testSize (float, optional): test size. Defaults to 0.2.

        Returns:
            sklearn.svm.SVC: SVC model
        """
        assert embeddingType in self.embedding, "Embedding embeddingType is not computed, please use computeWord2Vec() or computeGPT() to compute it"
        from sklearn.svm import SVC
        #train test split
        from sklearn.model_selection import train_test_split
        #from sklearn.model_selection import cross_val_score
        # get embedding
        embedding = self.embedding[embeddingType]
        # compute SVC
        svc = SVC(kernel=kernel, C=C, gamma=gamma)
        yList = [val for _, val in self.y.items()]
        X_train, X_test, y_train, y_test = train_test_split(embedding, yList, test_size=testSize, random_state=42)
        svc.fit(X_train, y_train)
        score = svc.score(X_test, y_test)
        print("SVC score : " + str(score))
        self.model["SVC"] = svc
        self.yPred["SVC"] = svc.predict(embedding)
        print("SVC model computed : self.model[\"SVC\"]")
        print("SVC prediction computed : self.y_pred[\"SVC\"]")
        return score
        
    def computeHDBSCAN(self, embeddingType = "word2vec", min_cluster_size = 2, min_samples = None, metric = "euclidean"):
        """Compute HDBSCAN model
        
        Args:
            embeddingType (str, optional): type of embedding. Defaults to "word2vec".
            min_cluster_size (int, optional): min cluster size. Defaults to 5.
            min_samples (int, optional): min samples. Defaults to None.
            metric (str, optional): metric. Defaults to "euclidean".

        Returns:
            hdbscan.HDBSCAN: HDBSCAN model
        """
        assert embeddingType in self.embedding, "Embedding embeddingType is not computed, please use computeWord2Vec() or computeGPT() to compute it"
        import hdbscan
        # get embedding
        embedding = self.embedding[embeddingType]
        # compute HDBSCAN
        hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, gen_min_span_tree=True)
        hdbscan.fit(embedding)
        self.model["HDBSCAN"] = hdbscan
        self.yPred["HDBSCAN"] = ["cluster_" + str(val) if val != -1 else "noise" for val in hdbscan.labels_]
        print("HDBSCAN model computed : self.model[\"HDBSCAN\"]")
        print("HDBSCAN prediction computed : self.y_pred[\"HDBSCAN\"]")

    def plotHDBSCANDendrogram(self, condensed = False): 
        """Plot HDBSCAN dendrogram
        
        Returns:
            plotly.graph_objects.Figure: dendrogram
        """
        assert "HDBSCAN" in self.model, "HDBSCAN model is not computed, please use computeHDBSCAN() to compute it"
        import numpy as np
        import matplotlib.pyplot as plt
        model = self.model["HDBSCAN"]
        import seaborn as sns
        # subplots ax

        if condensed:
            model.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
        else:
            model.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)

        
        

    def getAccuracy(self, model = "SVC"):
        """Get accuracy
        
        Args:
            model (str, optional): model. Defaults to "SVC".

        Returns:
            float: accuracy
        """
        assert model in self.model, "Model model is not computed, please use computeSVC() or computeHDBSCAN() to compute it"
        from sklearn.metrics import accuracy_score
        return accuracy_score(list(self.y.values()), self.yPred[model])