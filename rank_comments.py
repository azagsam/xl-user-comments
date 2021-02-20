import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from utils.helpers import load_sentences_from_file, load_sentences_from_AutoSentiNews, \
    load_sentences_from_cro_comments, load_sentences_from_cro_comments_vecernji

from utils.encoders import SentenceBERT, CMLM, LaBSE, LASER
import json

from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

import os

from utils.LexRank import degree_centrality_scores


def textrank(articles, encoders, reduction_methods, reduction_methods_params):
    """
    Description: Similarity between any two sentences is used as an equivalent to the web page transition probability
    """

    for enc_name, enc_model in encoders.items():
        # load sentence encoder
        print(enc_name)
        sentence_encoder = enc_model()

        for reduction_method_name in reduction_methods.keys():
            # load reduction model
            print(reduction_method_name)

            # fetch files
            for file in tqdm(os.scandir(articles)):
                with open(file.path) as f:
                    json_file = json.load(f)
                    article, ids = [], []
                    for comment_id, sents in json_file.items():
                        for s in sents:
                            article.append(s)
                            ids.append(comment_id)
                    assert len(article) == len(ids)

                # Compute sentence embeddings
                embeddings = sentence_encoder.encode_sentences(article)

                # Select dim reduction method
                if reduction_method_name != 'None':
                    red_method = reduction_methods[reduction_method_name](**reduction_methods_params[reduction_method_name])
                    embeddings = red_method.fit_transform(embeddings)

                # similarity matrix
                sim_mat = cosine_similarity(embeddings)
                np.fill_diagonal(sim_mat, 0)

                # rescale cosine score to [0, 1]
                scaler = MinMaxScaler(feature_range=(0, 1))
                sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings),
                                                                                         len(embeddings))
                np.fill_diagonal(sim_mat, 0)

                # calculate pagerank, monitor convergence
                try:
                    nx_graph = nx.from_numpy_array(sim_mat)
                    scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=500)

                    # get all scores for each comment
                    d = defaultdict(list)
                    for idx, (s , comment_id) in enumerate(zip(article, ids)):
                        d[comment_id].append(scores[idx])

                    # evaluate comment based on a function: max, mean, ...
                    scored_comment = {}
                    for key, value in d.items():
                        scored_comment[key] = (max(value), np.argmax(value))

                    # save into df with columns: article_id, comment_id, text, score
                    df = defaultdict(list)
                    for comment_id, score in scored_comment.items():
                        score, best_sentence_id = score
                        df['comment_id'].append(comment_id)
                        text = [f'<best_sentence> {s} </best_sentence>' if i == best_sentence_id else s for i, s in enumerate(json_file[comment_id])]
                        df['text'].append(' '.join(text))
                        df['score'].append(score)
                        df['article_id'].append(int(file.name.split('.')[0]))
                    df = pd.DataFrame(df)
                    df = df[['article_id', 'comment_id', 'text', 'score']]
                    df = df.round(4)

                    # get golden summary, append it to the end of the row
                    file_id = int(file.name.split('.')[0])
                    with open(f'data/kristina/cro_summaries/reference/vecernji_{file_id}.tgt') as g:
                        gold = ' '.join([line.strip() for line in g])

                    df.loc[-1] = [int(file.name.split('.')[0]), 'gold', gold, 1]

                    if not os.path.isfile('output/comments.csv'):
                        df.to_csv('output/comments.csv', mode='w', header=True, index=False)
                    else:
                        df.to_csv('output/comments.csv', mode='a', header=False, index=False)

                except nx.exception.PowerIterationFailedConvergence:
                    print(f'File {file.name} did not converge')


if __name__ == '__main__':
    encoders = {
        # 'SentenceBERT': SentenceBERT,
        'LaBSE': LaBSE,
        # 'CMLM': CMLM,
        # 'LASER': LASER  # Todo: NOTE: you have to specify language because the model uses different tokenizers
    }

    reduction_methods = {
        # 'pca': PCA,
        # 'umap': umap.UMAP,
        # 't-sne': TSNE,
        'None': None
    }

    reduction_methods_params = {
        'pca': {'n_components': 2},
        'umap': {'n_neighbors': 5, 'random_state': 42},  # check neighbors parameter
        # 't-sne': {'n_components': 2, 'perplexity': 30, 'random_state': 42},
        'None': None
    }

    # cluster_algo = {
    #     'kmeans': KMeans,
    #     'mshift': MeanShift,  # autodetermines number of clusters
    #     'gaussian_mixture': GaussianMixture #(n_components=3, covariance_type='full')
    # }

    # data
    articles = '/home/ales/Documents/RSDO/graph-based/data/vecernji_comments'

    # graph-based
    textrank(articles=articles, encoders=encoders, reduction_methods=reduction_methods, reduction_methods_params=reduction_methods_params)