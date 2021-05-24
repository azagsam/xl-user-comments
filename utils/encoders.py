from sentence_transformers import SentenceTransformer


class SentenceBERT:
    def __init__(self):
        self.model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    def encode_sentences(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)
