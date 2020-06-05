from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def visualize(dataset):
    representation, context, label = dataset[:, :-2], dataset[:, -2].astype(int), dataset[:, -1].astype(int)
    subset = np.where(label == 0)[0]
    representation, context = representation[subset], context[subset]
    # subset = np.where(context == 5)[0]
    # representation, label= representation[subset], label[subset]
    color_list = ["red", "blue", "yellow", "green", "purple", "black", "crimson", "orange", "deepskyblue", "brown"]
    tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=30, metric="euclidean", method="barnes_hut", init="random")
    tsne_rep = tsne.fit_transform(representation)
    plt.scatter(tsne_rep.T[0], tsne_rep.T[1], c=[color_list[i] for i in context])
    # plt.scatter(tsne_rep.T[0], tsne_rep.T[1], c=[color_list[i] for i in label])
    plt.show()