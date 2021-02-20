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

import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

import os

from utils.LexRank import degree_centrality_scores


def lexrank(articles, encoders, reduction_methods, reduction_methods_params):
    for enc_name, enc_model in encoders.items():
        # load sentence encoder
        print(enc_name)
        sentence_encoder = enc_model()

        for reduction_method_name in reduction_methods.keys():
            # load reduction model
            print(reduction_method_name)

            # create folders
            exp_name = f'{enc_name}_{reduction_method_name}_lexrank'
            print(exp_name)
            full_folder = os.path.join(output_dir, exp_name)

            if os.path.exists(full_folder):
                print('Folder exists ... continue with the next reduction method ... ')
                continue
            else:
                print('Experiment does not exist ... ')

            os.makedirs(full_folder)

            for file in tqdm(os.scandir(articles)):
                with open(file.path) as f:
                    article = [sent.strip() for sent in f.readlines()]

                # Compute the sentence embeddings
                embeddings = sentence_encoder.encode_sentences(article)

                # select dim reduction method
                if reduction_method_name != 'None':
                    red_method = reduction_methods[reduction_method_name](**reduction_methods_params[reduction_method_name])
                    embeddings = red_method.fit_transform(embeddings)

                # Compute the pair-wise cosine similarities
                if type(embeddings) == np.ndarray:
                    cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()
                else:
                    cos_scores = util.pytorch_cos_sim(embeddings.numpy(), embeddings.numpy())

                # rescale
                scaler = MinMaxScaler(feature_range=(0, 1))
                cos_scores = scaler.fit_transform(cos_scores.flatten().reshape(-1, 1)).reshape(len(embeddings),
                                                                                               len(embeddings))
                # Compute the centrality for each sentence
                centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

                # We argsort so that the first element is the sentence with the highest score
                most_central_sentence_indices = np.argsort(-centrality_scores)

                # Extract top n sentences as the summary
                full_path = os.path.join(full_folder, file.name)
                with open(full_path, 'w') as f:
                    # write 2 sentences
                    for s in range(2):
                        f.write(article[most_central_sentence_indices[s]])
                        f.write('\n')


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

            # create folders
            exp_name = f'{enc_name}_{reduction_method_name}_textrank'
            print(exp_name)
            full_folder = os.path.join(output_dir, exp_name)

            if os.path.exists(full_folder):
                print('Folder exists ... continue with the next reduction method ... ')
                continue
            else:
                print('Experiment does not exist ... ')

            os.makedirs(full_folder)

            for file in tqdm(os.scandir(articles)):
                with open(file.path) as f:
                    article = [sent.strip() for sent in f.readlines()]

                # Compute the sentence embeddings
                embeddings = sentence_encoder.encode_sentences(article)

                # select dim reduction method
                if reduction_method_name != 'None':
                    red_method = reduction_methods[reduction_method_name](**reduction_methods_params[reduction_method_name])
                    embeddings = red_method.fit_transform(embeddings)

                # similarity matrix
                sim_mat = cosine_similarity(embeddings)
                # np.fill_diagonal(sim_mat, 0)

                # rescale
                scaler = MinMaxScaler(feature_range=(0, 1))
                sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings),
                                                                                         len(embeddings))
                # np.fill_diagonal(sim_mat, 0)

                # calculate pagerank
                try:
                    nx_graph = nx.from_numpy_array(sim_mat)
                    scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=500)  # number of cycles to converge
                    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(article)), reverse=True)

                    # Extract top n sentences as the summary
                    full_path = os.path.join(full_folder, file.name)
                    with open(full_path, 'w') as f:
                        # write the first 2 sentences
                        for s in range(2):
                            f.write(ranked_sentences[s][1])
                            f.write('\n')

                except nx.exception.PowerIterationFailedConvergence:
                    print(f'File {file.name} did not converge')


def find_nearest_sentences(centroid, candidates, n=1, similarity_func='cosine_similarity'):
    """
    :parameter centroid: n-dim vector
    :parameter candidates: list of triplets (sentence, embedding)
    :parameter n: how many sentences to extract
    :parameter similarity_func: function for estimating similarity
    """

    nearest = []
    for can, emb in candidates:
        if type(emb) == np.ndarray:
            score = cosine_similarity(centroid.reshape(1, -1), emb.reshape(1, -1))
        else:  # convert to numpy
            score = cosine_similarity(centroid.reshape(1, -1), emb.numpy().reshape(1, -1))
        nearest.append((score, can))

    if similarity_func == 'cosine_similarity':
        nearest.sort(reverse=True)
    else:
        print('List has not been sorted. Corrected the code.')

    return [nearest[i][1] for i in range(n)]


def kmeans_cluster(articles, encoders, reduction_methods, reduction_methods_params, exp='kmeans'):
    for enc_name, enc_model in encoders.items():
        # load sentence encoder
        print(enc_name)
        sentence_encoder = enc_model()

        for reduction_method_name in reduction_methods.keys():
            # load reduction model
            print(reduction_method_name)

            # create folders
            exp_name = f'{enc_name}_{reduction_method_name}_{exp}'
            print(exp_name)
            full_folder = os.path.join(output_dir, exp_name)

            if os.path.exists(full_folder):
                print('Folder exists ... continue with the next reduction method ... ')
                continue
            else:
                print('Experiment does not exist ... ')

            os.makedirs(full_folder)

            for file in tqdm(os.scandir(articles)):
                with open(file.path) as f:
                    article = [sent.strip() for sent in f.readlines()]

                # Compute the sentence embeddings
                embeddings = sentence_encoder.encode_sentences(article)

                # select dim reduction method
                if reduction_method_name != 'None':
                    red_method = reduction_methods[reduction_method_name](**reduction_methods_params[reduction_method_name])
                    embeddings = red_method.fit_transform(embeddings)

                kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
                labels = list(kmeans.labels_)

                extracted = []
                for idx, centroid in enumerate(kmeans.cluster_centers_):
                    sent_embedding = [(sent, embedding) for sent, embedding, cluster in zip(article, embeddings, labels) if cluster == idx]
                    for sent in find_nearest_sentences(centroid, sent_embedding, n=1, similarity_func='cosine_similarity'):
                        extracted.append(sent)

                # Extract top n sentences as the summary
                full_path = os.path.join(full_folder, file.name)
                with open(full_path, 'w') as f:
                    for sent in extracted:
                        f.write(sent)
                        f.write('\n')


def gausian_mixture_cluster(articles, encoders, reduction_methods, reduction_methods_params, exp='gaussianmixture'):
    for enc_name, enc_model in encoders.items():
        # load sentence encoder
        print(enc_name)
        sentence_encoder = enc_model()

        for reduction_method_name in reduction_methods.keys():
            # load reduction model
            print(reduction_method_name)

            # create folders
            exp_name = f'{enc_name}_{reduction_method_name}_{exp}'
            print(exp_name)
            full_folder = os.path.join(output_dir, exp_name)

            if os.path.exists(full_folder):
                print('Folder exists ... continue with the next reduction method ... ')
                continue
            else:
                print('Experiment does not exist ... ')

            os.makedirs(full_folder)

            for file in tqdm(os.scandir(articles)):
                with open(file.path) as f:
                    article = [sent.strip() for sent in f.readlines()]

                # Compute the sentence embeddings
                embeddings = sentence_encoder.encode_sentences(article)

                # select dim reduction method
                if reduction_method_name != 'None':
                    red_method = reduction_methods[reduction_method_name](**reduction_methods_params[reduction_method_name])
                    embeddings = red_method.fit_transform(embeddings)

                gm = GaussianMixture(n_components=2, covariance_type='full').fit(embeddings)
                labels = gm.predict(embeddings)

                extracted = []
                for idx, centroid in enumerate(gm.means_):
                    sent_embedding = [(sent, embedding) for sent, embedding, cluster in zip(article, embeddings, labels) if cluster == idx]
                    for sent in find_nearest_sentences(centroid, sent_embedding, n=1, similarity_func='cosine_similarity'):
                        extracted.append(sent)

                # Extract top n sentences as the summary
                full_path = os.path.join(full_folder, file.name)
                with open(full_path, 'w') as f:
                    for sent in extracted:
                        f.write(sent)
                        f.write('\n')


if __name__ == '__main__':
    encoders = {
        'SentenceBERT': SentenceBERT,
        'LaBSE': LaBSE,
        'CMLM': CMLM,
        # 'LASER': LASER  # Todo: NOTE: you have to specify language because the model uses different tokenizers
    }

    reduction_methods = {
        'pca': PCA,
        'umap': umap.UMAP,
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
    articles = '/home/azagar/myfiles/graph-based/data/kristina/cro_summaries/article'
    output_dir = 'output/kristina-new'

    # graph-based
    textrank(articles=articles, encoders=encoders, reduction_methods=reduction_methods, reduction_methods_params=reduction_methods_params)
    # lexrank(articles=articles, encoders=encoders, reduction_methods=reduction_methods, reduction_methods_params=reduction_methods_params)

    # # clustering
    kmeans_cluster(articles=articles, encoders=encoders, reduction_methods=reduction_methods, reduction_methods_params=reduction_methods_params, exp='kmeans')
    gausian_mixture_cluster(articles, encoders, reduction_methods, reduction_methods_params, exp='gaussianmixture')

