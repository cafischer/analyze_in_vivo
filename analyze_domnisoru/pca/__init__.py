from sklearn.decomposition import PCA


def perform_PCA(data_centered, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data_centered)
    projection = pca.transform(data_centered)
    return projection, pca.components_, pca.explained_variance_ratio_