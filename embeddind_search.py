"""
    Модуль для поиска по эмбеддингам
"""
import pickle

from sklearn.neighbors import KDTree
import numpy as np

class EmbeddingSearcher:

    def __init__(self, embeds_file):
        self.embeds = pickle.load(open(embeds_file, 'rb'))
        self.num_embeds = len(embeds)
        embed_index = dict(zip(self.embeds.keys(), range(self.num_embeds)))
        inverted_index = {j: i for i, j in self.embed_index.items()}

        self.embed_matrix = np.vstack([embeds[k] for k in embed_index])

        self.kdt = KDTree(embed_matrix, leaf_size=30, metric='euclidean')

    def query(self, embedding):
        q = self.kdt.query(test_example.reshape(1, -1), k=5, return_distance=False)
        return inverted_index[sim_query[0][0]]


    def example(self):
        test_id = embed_index[list(self.embed_index.keys())[620]]

        test_example = embed_matrix[test_id]
        sim_query = self.query(test_example)

        return sim_query
