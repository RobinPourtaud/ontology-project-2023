def plotEmbedding(X,y, dim = 3):
    """Plot embedding in 2D or 3D
    Args:
        X (numpy.ndarray): 2D array of features
        y (numpy.ndarray): 1D array of labels
        dim (int, optional): 2D or 3D. Defaults to 3.
    """
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    import plotly.express as px

    # PCA
    pca = PCA(n_components=dim)
    result = pca.fit_transform(X)

    # Plot
    df = pd.DataFrame(result, columns = ['x', 'y', 'z'][:dim], index = X.index)
    df['label'] = y
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label')
    fig.show()

    