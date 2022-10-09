import numpy as np
from scipy import spatial
import random
from pprint import pprint
import pandas as pd


## Set a random seed for reproducibility
random.seed(1337)

## Generate some random vectors
embedding_01 = [random.uniform(-1, 1) for i in range(10)]
embedding_02 = [random.uniform(-1, 1) for i in range(10)]

## Cosine similarity.
## 1 - spatial.distance.cosine because higher = more similar
def cos_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)


## Euclidian distance
## You'll definitely want to figure out a way of normalizing this.  One way is probably dist/max_dist

# normalize euclidean distance
def euclidean_distance(a, b):
    return spatial.distance.euclidean(a, b) / max(a + b)


# try all distance functions from scipy.spatial.distance on two embeddings


def all_distances(a, b):
    # # inverse of covariance matrix of a an b
    # cov = np.linalg.inv(np.cov(a, b))
    # print(cov)

    return {
        "euclidean": euclidean_distance(a, b),
        "cosine": cos_similarity(a, b),
        "braycurtis": spatial.distance.braycurtis(a, b),
        "canberra": spatial.distance.canberra(a, b),
        "chebyshev": spatial.distance.chebyshev(a, b),
        "cityblock": spatial.distance.cityblock(a, b),
        "correlation": spatial.distance.correlation(a, b),
        "dice": spatial.distance.dice(a, b),
        "hamming": spatial.distance.hamming(a, b),
        "jaccard": spatial.distance.jaccard(a, b),
        "kulsinski": spatial.distance.kulsinski(a, b),
        # "mahalanobis": spatial.distance.mahalanobis(a, b, cov),
        "matching": spatial.distance.matching(a, b),
        "minkowski": spatial.distance.minkowski(a, b),
        "rogerstanimoto": spatial.distance.rogerstanimoto(a, b),
        "russellrao": spatial.distance.russellrao(a, b),
        # "seuclidean": spatial.distance.seuclidean(a, b),
        "sokalmichener": spatial.distance.sokalmichener(a, b),
        "sokalsneath": spatial.distance.sokalsneath(a, b),
        "sqeuclidean": spatial.distance.sqeuclidean(a, b),
        "yule": spatial.distance.yule(a, b),
    }


# print(embedding_01)
# print(embedding_02)
# pprint(all_distances(embedding_01, embedding_02))


mystery = pd.read_csv("data/challenge.csv")
cnn = pd.read_csv("data/cnn_samples.csv")
federal = pd.read_csv("data/federal_samples.csv")


def compute_similarities(df):

    df["cosine"] = df["embeddings"].apply(
        lambda x: cos_similarity(mystery["embeddings"], x)
    )
    df["euclidean"] = df["embeddings"].apply(
        lambda x: euclidean_distance(mystery["embeddings"], x)
    )

    return df.sort_values(by=["cosine", "euclidean"], ascending=False)


cnn = compute_similarities(cnn)
federal = compute_similarities(federal)


## See https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance for other distance measures you might be able to leverage.
