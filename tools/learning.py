def learning(X, y, kfold=5):
    """Step 6: Clustering and Classification

    Args:
        X (numpy.ndarray): 2D array of features
        y (numpy.ndarray): 1D array of labels

    Returns:
        tuple: (X, y)
    """

    ## Clustering HDBSCAN
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
    clusterer.fit(X)
    y_pred = clusterer.labels_

    ## Classification SVM
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    clf = SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, X, y, cv=kfold)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return {
        "X": X,
        "y_pred": y_pred, 
        "scores": scores, 
        "hdbscan": clusterer,
        "SVM": clf
    }
    
