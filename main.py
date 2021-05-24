from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from utils.encoders import SentenceBERT


def summarize(comments, n):
    sentence_encoder = SentenceBERT()

    # Compute the sentence embeddings
    embeddings = sentence_encoder.encode_sentences(comments)

    # similarity matrix
    sim_mat = cosine_similarity(embeddings)

    # rescale
    scaler = MinMaxScaler(feature_range=(0, 1))
    sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))

    # calculate pagerank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=500)  # number of cycles to converge
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(comments)), reverse=True)
    return [sent for score, sent in ranked_sentences[:n]]


if __name__ == '__main__':
    # load example_comments.txt (assuming pre-tokenized examples on a sentence level)
    with open('example_comments.txt') as f:
        comments = [sent.strip() for sent in f.readlines()]

    # run main
    summarize(comments, n=2)

