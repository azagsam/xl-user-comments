import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import tensorflow_text
from utils.utils_labse import get_model, create_input
import bert

from laserembeddings import Laser


class SentenceBERT:
    def __init__(self):
        self.model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    def encode_sentences(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)


class CMLM:
    def __init__(self):
        self.preprocessor = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/1")
            # "file:///home/ales/Documents/RSDO/graph-based/models/CMLM/universal-sentence-encoder-cmlm_multilingual-preprocess_1")
        self.encoder = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1")
            # "file:///home/ales/Documents/RSDO/graph-based/models/CMLM/universal-sentence-encoder-cmlm_multilingual-base-br_1")


    def encode_sentences(self, sentences):
        return self.encoder(self.preprocessor(sentences))["default"]


class LaBSE:
    def __init__(self):
        self.max_seq_length = 64
        self.model, self.layer = get_model(
            # in case of troubles use "file:///home/ales/Documents/RSDO/graph-based/models/LaBSE"
            # model_url="file:///home/ales/Documents/RSDO/graph-based/models/LaBSE", max_seq_length=self.max_seq_length)
            model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=self.max_seq_length)
        self.vocab_file = self.layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = bert.bert_tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)

    def encode_sentences(self, sentences):
        input_ids, input_mask, segment_ids = create_input(
            sentences, self.tokenizer, self.max_seq_length)
        return self.model([input_ids, input_mask, segment_ids])


class LASER:
    def __init__(self):
        self.model = Laser()

    def encode_sentences(self, sentences, lang):
        return self.model.embed_sentences(sentences, lang)