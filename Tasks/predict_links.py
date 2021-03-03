#!/usr/bin/env python3
# encoding: UTF-8

"""
Description:
    Use node2vec for link prediction.
"""
import random
import os
import pickle
import node2vec
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from preprocessing import mask_test_edges
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, f1_score
from clustering import cluster
import itertools

NETWORK_DIR = "pickle"
PICKLE_FILE = "adj_feat_Graph.pkl"
ID_MAP_FILE = "id_map.pkl"


# All the parameters need to be tuned are here.
emb_paras = {"p": 8,
             "q": 0.5,
             "win_size": 10,
             "num_walks": 10,
             "walk_length": 20,
             "dimension": 55,
             "iter": 1,
             "rocfile": "Plots/roc" + str(1) + ".png",
             "result_file_path": "results/parameters_" + str(1) + ".png"}

tuning_para = {"n_clusters": 6,
               "n_init": 10,
               "max_iter": 300,
               "random_state": 42,
               "tsne_perplexity": 80,
               "fig_path": "Plots/cluster" + str(1) + ".txt"}


def main():
    p = [0.2]
    q = [0.2]
    win_size = [5]
    num_walks = [10]
    walk_length = [10]
    dimension = [50]
    iter = [1]

    n_clusters = [8]
    n_init = [7]
    max_iter = [400]
    random_state = [42]
    tsne_perplexity = [80]

    """
    p = [0.2, 0.6, 0.8, 2, 4, 6]
    q = [0.2, 0.6, 0.8, 2, 4, 6]
    win_size = [5, 10, 15]
    num_walks = [10, 20, 30]
    walk_length = [10, 20, 30]
    dimension = [50, 100, 128]
    iter = [1, 2]

    n_clusters = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    n_init = [7, 14]
    max_iter = [200, 300, 400]
    random_state = [42]
    tsne_perplexity = [40, 60, 80, 100]
"""

    n = list(itertools.product(*[p, q, win_size, num_walks, walk_length, dimension, iter]))
    m = list(itertools.product(*[n_clusters, n_init, max_iter, random_state, tsne_perplexity]))
    Version = 1000000
    for i in n:
        
        emb_paras["p"] =  i[0]
        emb_paras["q"] = i[1]
        emb_paras["win_size"] = i[2]
        emb_paras["num_walks"] = i[3]
        emb_paras["walk_length"] = i[4]
        emb_paras["dimension"] = i[5]
        emb_paras["iter"] = i[6]


        for j in m:
            tuning_para["n_clusters"] = j[0]
            tuning_para["n_init"] = j[1]
            tuning_para["max_iter"] = j[2]
            tuning_para["random_state"] = j[3]
            tuning_para["tsne_perplexity"] = j[4]
            tuning_para["fig_path"] = "Plots/cluster_" + str(Version) + ".png"
            emb_paras["rocfile"] = "Plots/roc_" + str(Version) + ".png"
            emb_paras["result_file_path"] = "results/parameters_" + str(Version) + ".txt"
            link_pred_emb(**emb_paras)
            Version += 1






def link_pred_emb(p=8, q=0.5, win_size=10,num_walks=10, walk_length=20, dimension=55, iter=1, rocfile="Plots/roctest.png", result_file_path = "results/parameters.txt") -> None:
    """The main function. Link prediction is done here."""

    # Load pickled (adj, feat) tuple
    with open(os.path.join(NETWORK_DIR, PICKLE_FILE), "rb") as file:
        adj, features = pickle.load(file)
    with open(os.path.join(NETWORK_DIR, ID_MAP_FILE), "rb") as file:
        id_map = pickle.load(file)


    g = nx.Graph(adj)  # Recreate graph using node indices (0 to num_nodes-1)
    # Draw the network
    # nx.draw_networkx(g, with_labels=False, node_size=50, node_color="r")
    # plt.show()

    # Preprocessing (train/test split)
    np.random.seed(0)  # make sure train-test split is consistent
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    # Perform train-test split
    (
        adj_train,
        train_edges,
        train_edges_false,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges(adj_sparse, test_frac=0.3, val_frac=0.1)

    # new graph object with only non-hidden edges
    g_train = nx.from_scipy_sparse_matrix(adj_train)

    # Inspect train/test split
    print("Total nodes:", adj_sparse.shape[0])

    # adj is symmetric, so nnz (num non-zero) = 2 * num_edges
    print("Total edges:", int(adj_sparse.nnz / 2))
    print("Training edges (positive):", len(train_edges))
    print("Training edges (negative):", len(train_edges_false))
    print("Validation edges (positive):", len(val_edges))
    print("Validation edges (negative):", len(val_edges_false))
    print("Test edges (positive):", len(test_edges))
    print("Test edges (negative):", len(test_edges_false))

    # Train node2vec (Learn Node Embeddings)

    # node2vec settings
    # NOTE: When p = q = 1, this is equivalent to DeepWalk

    # P = 5   # Return hyperparameter
    # Q = 0.65  # In-out hyperparameter
    # WINDOW_SIZE = 10  # Context size for optimization
    # NUM_WALKS = 5  # Number of walks per source
    # WALK_LENGTH = 5  # Length of walk per source
    # DIMENSIONS = 128  # Embedding dimension
    # DIRECTED = False  # Graph directed/undirected
    # WORKERS = 8  # Num. parallel workers
    # ITER = 1  # SGD epochs

    P = p   # Return hyperparameter
    Q = q  # In-out hyperparameter
    WINDOW_SIZE = win_size  # Context size for optimization
    NUM_WALKS = num_walks  # Number of walks per source
    WALK_LENGTH = walk_length  # Length of walk per source
    DIMENSIONS = dimension  # Embedding dimension
    DIRECTED = False  # Graph directed/undirected
    WORKERS = 8  # Num. parallel workers
    ITER = iter  # SGD epochs

    # Preprocessing, generate walks

    # create node2vec graph instance
    g_n2v = node2vec.Graph(g_train, DIRECTED, P, Q)
    g_n2v.preprocess_transition_probs()
    walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
    walks = [list(map(str, walk)) for walk in walks]

    # Train skip-gram model
    model = Word2Vec(
        walks,
        size=DIMENSIONS,
        window=WINDOW_SIZE,
        min_count=0,
        sg=1,
        workers=WORKERS,
        iter=ITER,
    )

    # Store embeddings mapping
    emb_mappings = model.wv
    model.wv.save_word2vec_format('Neo-Emb-2.emd')

    # Create node embeddings matrix (rows = nodes, columns = embedding features)
    emb_list = []
    for node_index in range(0, adj_sparse.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    def get_edge_embeddings(edge_list):
        """
        Generate bootstrapped edge embeddings (as is done in node2vec paper)
        Edge embedding for (v1, v2) = hadamard product of node embeddings for
        v1, v2.
        """
        embs = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            embs.append(edge_emb)
        embs = np.array(embs)
        return embs

    # Train-set edge embeddings
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate(
        [pos_train_edge_embs, neg_train_edge_embs]
    )

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate(
        [np.ones(len(train_edges)), np.zeros(len(train_edges_false))]
    )

    # Val-set edge embeddings, labels
    pos_val_edge_embs = get_edge_embeddings(val_edges)
    neg_val_edge_embs = get_edge_embeddings(val_edges_false)
    val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
    val_edge_labels = np.concatenate(
        [np.ones(len(val_edges)), np.zeros(len(val_edges_false))]
    )

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate(
        [np.ones(len(test_edges)), np.zeros(len(test_edges_false))]
    )

    # Train logistic regression classifier on train-set edge embeddings
    #edge_classifier = LogisticRegression(random_state=0)
    edge_classifier = RandomForestClassifier(max_depth=10, random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    val_roc = roc_auc_score(val_edge_labels, val_preds)
    val_ap = average_precision_score(val_edge_labels, val_preds)

    # Predicted edge scores: probability of being of class "1" (real edge)
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    test_roc = roc_auc_score(test_edge_labels, test_preds)
    test_ap = average_precision_score(test_edge_labels, test_preds)

    result_file = open(result_file_path, "w")
    for para, value in emb_paras.items():
        if para != "roc_file":
            result_file.write(para + "  :  " + str(value))
            result_file.write("\n")

    for para, value in tuning_para.items():
        if para != "fig_path":
            result_file.write(para + "  :  " + str(value))
            result_file.write("\n")



    result_file.write("node2vec Validation ROC score: " + str(val_roc))
    result_file.write("\n")
    result_file.write("node2vec Validation AP score: " + str(val_ap))
    result_file.write("\n")
    result_file.write("node2vec Test ROC score: " + str(test_roc))
    result_file.write("\n")
    result_file.write("node2vec Test AP score: " + str(test_ap))
    result_file.write("\n")
    silhouette_score, purity_score = cluster(**tuning_para)
    result_file.write("silhouette score:" + str(silhouette_score))
    result_file.write("\n")
    result_file.write("purity score:" + str(purity_score))
    result_file.write("\n")
    result_file.close()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(test_edge_labels, test_preds)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='%s (area = %0.2f)' % ('RF', roc_auc))
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('test')
    plt.legend(loc='lower right')

    plt.savefig(rocfile)
    plt.close()


if __name__ == "__main__":
    main()
    print("Finished")