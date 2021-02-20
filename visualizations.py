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
    load_sentences_from_cro_comments, scale_centrality_scores, \
    get_context_for_sentences

from utils.encoders import SentenceBERT, CMLM, LaBSE, LASER
import json

import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

import os

from utils.LexRank import degree_centrality_scores


def plot_clustering_gm(sentences, encoder, rec_method_name, reduction_method, reduction_method_params, cluster_params):
    print(encoder)
    sentence_encoder = encoder()

    # load sentences
    embeddings = sentence_encoder.encode_sentences(sentences)

    # reduce dimensionality
    print(reduction_method)
    rm = reduction_method(**reduction_method_params)
    reduced_dim_embeddings = rm.fit_transform(embeddings)

    gm = GaussianMixture(**cluster_params).fit(reduced_dim_embeddings)
    clustering = gm.predict(reduced_dim_embeddings)

    # plot the points
    xs = [x for x, _ in reduced_dim_embeddings]
    ys = [y for _, y in reduced_dim_embeddings]
    labels = [f'Cluster {c}' for c in clustering]

    df = pd.DataFrame(
        {
            'x': xs,
            'y': ys,
            'cluster': labels,
            'sentences': sentences
        }
    )

    fig = px.scatter(df,
                     x='x',
                     y='y',
                     hover_name='sentences',
                     color='cluster',
                     title=f'<b>Dimensionality reduced by {rec_method_name}, colored with GaussianMixture File{i}</b>'
                     )

    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.add_trace(
        go.Scatter(
            x=[x for x, _ in gm.means_],
            y=[y for _, y in gm.means_],
            showlegend=False,
            hovertext=list(range(len(gm.means_))),
            mode='markers',
            marker=dict(
                color='Black',
                size=16,
                symbol='x',
            )
        )
    )

    fig.show()


def plot_clustering_gm_show_text(sentences, encoder, rec_method_name, reduction_method, reduction_method_params, cluster_params):
    print(encoder)
    sentence_encoder = encoder()

    # load sentences
    embeddings = sentence_encoder.encode_sentences(sentences)

    # reduce dimensionality
    print(reduction_method)
    rm = reduction_method(**reduction_method_params)
    reduced_dim_embeddings = rm.fit_transform(embeddings)

    gm = GaussianMixture(**cluster_params).fit(reduced_dim_embeddings)
    clustering = gm.predict(reduced_dim_embeddings)

    # plot the points
    xs = [x for x, _ in reduced_dim_embeddings]
    ys = [y for _, y in reduced_dim_embeddings]
    labels = [f'Cluster {c}' for c in clustering]

    df = pd.DataFrame(
        {
            'x': xs,
            'y': ys,
            'cluster': labels,
            'sentences': sentences
        }
    )

    fig = px.scatter(df,
                     x='x',
                     y='y',
                     text='sentences',
                     color='cluster',
                     title=f'<b>Dimensionality reduced by {rec_method_name}, colored with GaussianMixture File{i}</b>'
                     )

    fig.update_traces(textposition='bottom center')

    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'),
                      )

    fig.add_trace(
        go.Scatter(
            x=[x for x, _ in gm.means_],
            y=[y for _, y in gm.means_],
            showlegend=False,
            hovertext=list(range(len(gm.means_))),
            mode='markers',
            marker=dict(
                color='Black',
                size=16,
                symbol='x',
            )
        )
    )

    fig.show()


def plot_clustering_kmeans(sentences, encoder, rec_method_name, reduction_method, reduction_method_params, cluster_params):
    print(encoder)
    sentence_encoder = encoder()

    # load sentences
    embeddings = sentence_encoder.encode_sentences(sentences)

    # reduce dimensionality
    print(reduction_method)
    rm = reduction_method(**reduction_method_params)
    reduced_dim_embeddings = rm.fit_transform(embeddings)

    kmeans = KMeans(**cluster_params).fit(reduced_dim_embeddings)

    # plot the points
    xs = [x for x, _ in reduced_dim_embeddings]
    ys = [y for _, y in reduced_dim_embeddings]
    labels = [f'Cluster {c}' for c in kmeans.labels_]

    df = pd.DataFrame(
        {
            'x': xs,
            'y': ys,
            'cluster': labels,
            'sentences': sentences
        }
    )

    fig = px.scatter(df,
                     x='x',
                     y='y',
                     hover_name='sentences',
                     color='cluster',
                     title=f'<b>Dimensionality reduced by {rec_method_name}, colored with KMeans File{i} </b>'
                     )

    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.add_trace(
        go.Scatter(
            x=[x for x, _ in kmeans.cluster_centers_],
            y=[y for _, y in kmeans.cluster_centers_],
            showlegend=False,
            hovertext=list(range(len(kmeans.cluster_centers_))),
            mode='markers',
            marker=dict(
                color='Black',
                size=16,
                symbol='x',
            )
        )
    )

    fig.show()


def texrank_barplot(sentences, encoder, rec_method_name, reduction_method, reduction_method_params):
    print(encoder)
    sentence_encoder = encoder()

    # load sentences
    embeddings = sentence_encoder.encode_sentences(sentences)

    # # reduce dimensionality
    # print(reduction_method)
    # if rec_method_name != 'None':
    #     rm = reduction_method(**reduction_method_params)
    #     embeddings = rm.fit_transform(embeddings)

    # similarity matrix
    sim_mat = cosine_similarity(embeddings)
    np.fill_diagonal(sim_mat, 0)

    # rescale
    scaler = MinMaxScaler(feature_range=(0, 1))
    sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))
    np.fill_diagonal(sim_mat, 0)

    # calculate pagerank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph, max_iter=500)  # number of cycles to converge
    # ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    score_list = [scores[sent_idx] for sent_idx in range(len(sentences))]

    df = pd.DataFrame(
        {
            'scores': score_list,
            'sentences': sentences,
        }
    )

    fig = px.bar(df,
                 y=list(range(len(sentences))),
                 x='scores',
                 text='sentences',
                 orientation='h',
                 color='scores',
                 color_continuous_scale='OrRd',
                 title=f'<b>TextRank - Bar plot File{i}</b>'
                 )
    fig.update_traces(  # texttemplate='%{text:.2s}',
        textposition='inside')
    fig.update_yaxes(autorange="reversed",
                     )
    fig.show()


def lexrank_barplot(sentences, encoder, rec_method_name, reduction_method, reduction_method_params, q):
    print(encoder)
    sentence_encoder = encoder()

    # load sentences
    embeddings = sentence_encoder.encode_sentences(sentences)

    # # reduce dimensionality
    # print(reduction_method)
    # if rec_method_name != 'None':
    #     rm = reduction_method(**reduction_method_params)
    #     embeddings = rm.fit_transform(embeddings)

    # Compute the pair-wise cosine similarities
    cos_scores = util.pytorch_cos_sim(embeddings.numpy(), embeddings.numpy())

    # rescale
    scaler = MinMaxScaler(feature_range=(0, 1))
    cos_scores = scaler.fit_transform(cos_scores.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))

    # Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=0.2)

    # # scale for visualization purposes
    # centrality_scores = scale_centrality_scores(centrality_scores, q=q)

    df = pd.DataFrame(
        {
            'centrality': centrality_scores,
            'sentences': sentences,
        }
    )

    fig = px.bar(df,
                 y=list(range(len(sentences))),
                 x='centrality',
                 text='sentences',
                 orientation='h',
                 color='centrality',
                 color_continuous_scale='OrRd',
                 title=f'<b>LexRank - Bar plot File{i}</b>'
                 )
    fig.update_traces(textposition='inside')
    fig.update_yaxes(autorange="reversed")
    fig.show()


def lexrank_nodes_edges(sentences, encoder, rec_method_name, reduction_method, reduction_method_params, q):
    print(encoder)
    sentence_encoder = encoder()

    # load sentences
    embeddings = sentence_encoder.encode_sentences(sentences)

    # Compute the pair-wise cosine similarities
    cos_scores = util.pytorch_cos_sim(embeddings.numpy(), embeddings.numpy())

    # rescale
    scaler = MinMaxScaler(feature_range=(0, 1))
    cos_scores = scaler.fit_transform(cos_scores.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))

    # Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=0.2)

    # scale for visualization purposes
    if q:
        centrality_scores = scale_centrality_scores(centrality_scores, q=q)

    # We argsort so that the first element is the sentence with the highest score
    # most_central_sentence_indices = np.argsort(-centrality_scores)

    # reduce dimensionality
    print(rec_method_name)
    assert rec_method_name != 'None'
    rm = reduction_method(**reduction_method_params)
    pos = rm.fit_transform(embeddings)

    # get weights
    weights = cos_scores
    np.fill_diagonal(weights, 0)

    G = nx.from_numpy_array(weights)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if weights[edge[0], edge[1]] > 0.5:
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.75,
                  color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Reds',
            # reversescale=True,
            color=[],
            size=[s * 10 for s in centrality_scores],
            colorbar=dict(
                thickness=15,
                title='Centrality Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    node_adjacencies = []
    node_text = []
    for node, weight in enumerate(centrality_scores):
        node_adjacencies.append(weight)
        node_text.append(sentences[node])

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'<b>LexRank Summarization File{i}</b>',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        # annotations=[ dict(
                        #     #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'>
                        #     # https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        #     showarrow=False,
                        #     xref="paper", yref="paper",
                        #     x=0.005, y=-0.002 ) ],
                        # xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        # yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    ))
    fig.show()


def textrank_nodes_edges(sentences, encoder, rec_method_name, reduction_method, reduction_method_params, q):
    print(encoder)
    sentence_encoder = encoder()

    # load sentences
    embeddings = sentence_encoder.encode_sentences(sentences)

    # # reduce dimensionality
    # print(reduction_method)
    # if rec_method_name != 'None':
    #     rm = reduction_method(**reduction_method_params)
    #     embeddings = rm.fit_transform(embeddings)

    # similarity matrix
    sim_mat = cosine_similarity(embeddings)
    np.fill_diagonal(sim_mat, 0)

    # rescale
    scaler = MinMaxScaler(feature_range=(0, 1))
    sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))
    np.fill_diagonal(sim_mat, 0)

    # calculate pagerank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph, max_iter=500)  # number of cycles to converge
    # ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    score_list = [scores[sent_idx] for sent_idx in range(len(sentences))]

    # reduce dimensionality
    print(rec_method_name)
    assert rec_method_name != 'None'
    rm = reduction_method(**reduction_method_params)
    pos = rm.fit_transform(embeddings)

    # get weights
    weights = sim_mat
    centrality_scores = np.array(score_list)
    if q:
        centrality_scores = scale_centrality_scores(centrality_scores, q=q)
    np.fill_diagonal(weights, 0)

    G = nx.from_numpy_array(weights)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if weights[edge[0], edge[1]] > 0.8:
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1,
                  color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Greens',
            # reversescale=True,
            color=[],
            size=[s * 10 for s in centrality_scores],
            colorbar=dict(
                thickness=15,
                title='Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    # wrap a sentence with neighbour sentences
    context = get_context_for_sentences(sentences)
    node_adjacencies = []
    node_text = []
    for node, weight in enumerate(centrality_scores):
        node_adjacencies.append(weight)
        node_text.append(context[node])

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'<b>TextRank visualization with {rec_method_name} File{i}</b>',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        # annotations=[ dict(
                        #     #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'>
                        #     # https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        #     showarrow=False,
                        #     xref="paper", yref="paper",
                        #     x=0.005, y=-0.002 ) ],
                        # xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        # yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    ))
    fig.show()


def lexrank_nodes_edges_show_text(sentences, encoder, rec_method_name, reduction_method, reduction_method_params, q):
    print(encoder)
    sentence_encoder = encoder()

    # load sentences
    embeddings = sentence_encoder.encode_sentences(sentences)

    # Compute the pair-wise cosine similarities
    cos_scores = util.pytorch_cos_sim(embeddings.numpy(), embeddings.numpy())

    # rescale
    scaler = MinMaxScaler(feature_range=(0, 1))
    cos_scores = scaler.fit_transform(cos_scores.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))

    # Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=0.2)

    # scale for visualization purposes
    if q:
        centrality_scores = scale_centrality_scores(centrality_scores, q=q)

    # We argsort so that the first element is the sentence with the highest score
    # most_central_sentence_indices = np.argsort(-centrality_scores)

    # reduce dimensionality
    print(rec_method_name)
    assert rec_method_name != 'None'
    rm = reduction_method(**reduction_method_params)
    pos = rm.fit_transform(embeddings)

    # get weights
    weights = cos_scores
    np.fill_diagonal(weights, 0)

    G = nx.from_numpy_array(weights)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if weights[edge[0], edge[1]] > 0.5:
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.75,
                  color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    important_sentences = [sent if weight > 8.0 else 'X' for sent, weight in zip(sentences, centrality_scores)]
    print(important_sentences)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        # mode='markers',
        # hoverinfo='text',
        mode="markers+text",
        name="Markers and Text",
        text=important_sentences,
        textposition="bottom center",
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Reds',
            # reversescale=True,
            color=[],
            size=[s * 10 for s in centrality_scores],
            colorbar=dict(
                thickness=15,
                title='Centrality Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    node_adjacencies = []
    node_text = []
    for node, weight in enumerate(centrality_scores):
        node_adjacencies.append(weight)
        if weight > 8.0:
            node_text.append(sentences[node])
        else:
            node_text.append('')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'<b>LexRank Summarization File{i}</b>',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        # annotations=[ dict(
                        #     #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'>
                        #     # https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        #     showarrow=False,
                        #     xref="paper", yref="paper",
                        #     x=0.005, y=-0.002 ) ],
                        # xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        # yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    ))
    fig.show()


def load_sentences(name, **kwargs):
    if name == 'single_file':
        return load_sentences_from_file(file=kwargs['file'])
    if name == 'asn':
        return load_sentences_from_AutoSentiNews(idx=kwargs['idx'])
    if name == 'cro_comments':
        return load_sentences_from_cro_comments(idx=kwargs['idx'])


if __name__ == '__main__':
    # Sentence encoders and dimensionality reduction methods
    encoders = {
        'SentenceBERT': SentenceBERT,
        'CMLM': CMLM,
        'LaBSE': LaBSE,
        # 'LASER': LASER  # Todo: NOTE: you have to specify language because the model uses different tokenizers
    }

    reduction_methods = {
        'pca': PCA,
        'umap': umap.UMAP,
        't-sne': TSNE,
        'None': None
    }

    reduction_methods_params = {
        'pca': {'n_components': 2},
        'umap': {'n_neighbors': 5, 'random_state': 42},  # check neighbors parameter
        't-sne': {'n_components': 2, 'perplexity': 30, 'random_state': 42},
        'None': None
    }

    cluster_params = {
        'kmeans': {'n_clusters': 3, 'random_state': 0},
        'gaussian_mixture': {'n_components': 3, 'covariance_type': 'full'}
    }

    # import data
    for i in range(100):
        # sentences = load_sentences(name='single_file', file='data/cluster-translations.txt')
        sentences = load_sentences(name='cro_comments_vecernji', idx=i)
        # if len(sentences) < 10 or len(sentences) > 30:
        #     continue

        # plot clustering
        enc_name = 'SentenceBERT'
        rec_method_name = 'None'

        # # # TODO 1/12/21 Clustering
        # plot_clustering_gm(sentences, encoders[enc_name],
        #                    rec_method_name,
        #                    reduction_methods[rec_method_name],
        #                    reduction_methods_params[rec_method_name],
        #                    cluster_params['gaussian_mixture'])
        #
        # plot_clustering_gm_show_text(sentences, encoders[enc_name],
        #                    rec_method_name,
        #                    reduction_methods[rec_method_name],
        #                    reduction_methods_params[rec_method_name],
        #                    cluster_params['gaussian_mixture'])

        # plot_clustering_kmeans(sentences, encoders[enc_name],
        #                        rec_method_name,
        #                        reduction_methods[rec_method_name],
        #                        reduction_methods_params[rec_method_name],
        #                        cluster_params['kmeans'])


        # # # # TODO 1/12/21 Barplots
        texrank_barplot(sentences, encoders[enc_name],
                       rec_method_name,
                       reduction_methods[rec_method_name],
                       reduction_methods_params[rec_method_name])

        # lexrank_barplot(sentences, encoders[enc_name],
        #                rec_method_name,
        #                reduction_methods[rec_method_name],
        #                reduction_methods_params[rec_method_name],
        #                 q=0.5)  # centality score scaler

        # # TODO 1/12/21 Nodes + Edges
        # lexrank_nodes_edges(sentences, encoders[enc_name],
        #                 rec_method_name,
        #                 reduction_methods[rec_method_name],
        #                 reduction_methods_params[rec_method_name],
        #                 q=0.5)  # score scaler
        # lexrank_nodes_edges_show_text(sentences, encoders[enc_name],
        #                 rec_method_name,
        #                 reduction_methods[rec_method_name],
        #                 reduction_methods_params[rec_method_name],
        #                 q=0.5)  # score scaler
        #
        # textrank_nodes_edges(sentences, encoders[enc_name],
        #                 rec_method_name,
        #                 reduction_methods[rec_method_name],
        #                 reduction_methods_params[rec_method_name],
        #                 q=0.25)  # score scaler
