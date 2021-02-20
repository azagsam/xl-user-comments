import json
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_sentences_from_file(file):
    with open(file) as f:
        sentences = [line.strip() for line in f.readlines()]
    return sentences


def load_sentences_from_AutoSentiNews(idx):
    with open(f'/home/ales/Documents/Extended/datasets/ASN/output/json-cased/asn-sample/{idx}.json') as f:
        data = json.load(f)
        sentences = data['abstract'] + data['article']
    return sentences


def load_sentences_from_styria(idx):
    with open(f'/media/ales/Extended/datasets/STYRIA/output/json/test/{idx}.json') as f:
        data = json.load(f)
        sentences = data['abstract'] + data['article']
    return sentences


def load_sentences_from_cro_comments(idx):
    with open(f'/home/ales/Documents/Extended/datasets/CRO-COMMENTS/output/cro-stories-json-subsample/test/{idx}.json') \
            as f:
        data = json.load(f)
        sentences = data['article']
    return sentences


def load_sentences_from_cro_comments_vecernji(idx):
    files = list(os.scandir('/home/ales/Documents/RSDO/graph-based/data/kristina/cro_summaries/article'))
    with open(files[idx].path) as f:
        lines = [l.strip() for l in f.readlines()]
    return lines


def get_context_for_sentences(sentences):
    context = []
    for idx, sent in enumerate(sentences):
        if idx == 0:
            context.append('LEAD : ' + sent)
        elif idx == len(sentences) - 1:
            context.append('END : ' + sent)
            break
        else:
            p = sentences[idx - 1] + '<br>' + str(idx) + ' : ' + sent + '<br>' + sentences[idx + 1]
            context.append(p)
    return context


def scale_centrality_scores(centrality_scores, q=0.1):
    scaler = MinMaxScaler(feature_range=(1, 10))
    scaler.fit(centrality_scores[centrality_scores >= np.quantile(centrality_scores, q=q)].reshape(-1, 1))
    centrality_scores = scaler.transform(centrality_scores.reshape(-1, 1))
    centrality_scores = np.where(centrality_scores < 0, 0.5, centrality_scores).ravel()
    return centrality_scores
