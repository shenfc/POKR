import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


NETWORK_DIR = "pickle"
ID_MAP_FILE = "id_map.pkl"
LABEL_MAP_FILE = "data/dictionary.csv"
NEW_LABEL_MAP_FILE = "data/new_dictionary.csv"
ID_TO_LABEL_FILE = "data/id_to_label.csv"
# !!!!!!!! PLease remeber to change the name of ID_TO_LABEL_NAME_FILE !!!!!!!!!!!!!!
# EVERYTIME THERE IS A NEW CLUSTER
# OTHER WISE IT WILL OVERWRITE THE PREVIOUS FILE
ID_TO_LABEL_NAME_FILE = "data/id_to_labe_name.csv"


tuning_para = {"n_clusters": 6,
               "n_init": 10,
               "max_iter": 300,
               "random_state": 42,
               "tsne_perplexity": 80,
               "fig_path": "Plots/cluster" + str(1) + ".png"}


def cluster(n_clusters=5, n_init=10, max_iter=300, random_state=42, tsne_perplexity=80, fig_path="test.png"):
    X = []
    y = []
    with open("Neo-Emb-2.emd") as f:
        next(f)
        for line in f:
            splits = line.strip().split()
            label = splits[0]
            vec = [float(v) for v in splits[1:]]
            X.append(vec)

    kmeans = KMeans(
        init="random",
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state, )

    pred = kmeans.fit_predict(X)

    # print(kmeans.labels_[:50])
    # print(kmeans.inertia_)

    df = build_df_for_purity(ID_TO_LABEL_FILE, ID_TO_LABEL_NAME_FILE, pred)
    purity_score = purity(df)
    np.random.seed(0)

    tsne = TSNE(n_components=2, perplexity=tsne_perplexity)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10))

    for i in range(1, len(X_tsne), 1):
        # print(X_tsne[i][0])
        # print(X_tsne[i][1])
        # print(pred[i])

        plt.scatter(X_tsne[i][0], X_tsne[i][1], s=60)

    plt.savefig(fig_path)
    plt.close()
    return silhouette_score(X, pred), purity_score

# Generate id to label file and save it as a csv file. ONLY RUN ONCE
def generate_id_to_label():
    with open(os.path.join(NETWORK_DIR, ID_MAP_FILE), "rb") as file:
        id_map = pickle.load(file)
    label_dict = pd.read_csv(LABEL_MAP_FILE, header=None)
    label_dict[2] = label_dict[2].astype('category')
    label_dict['label_cat'] = label_dict[2].cat.codes
    label_dict.columns = ["neo4j_id", "Name", "Label", "Label_Cat"]
    label_dict.to_csv(NEW_LABEL_MAP_FILE)
    new_label_list = []
    name_label_list = []
    for i in range(2808):
        neo4j_id = id_map[i]
        l = label_dict[label_dict["neo4j_id"] == neo4j_id]["Label_Cat"].values[0]
        label = label_dict[label_dict["neo4j_id"] == neo4j_id]["Label"].values[0]
        name = label_dict[label_dict["neo4j_id"] == neo4j_id]["Name"].values[0]
        nl = str(label) + "_" + str(name)
        name_label_list.append(nl)
        new_label_list.append(l)
    id_list = list(range(2808))
    df = pd.DataFrame({'id': id_list, 'label_cat': new_label_list})
    df.to_csv(ID_TO_LABEL_FILE)
    df1 = pd.DataFrame({'id': id_list, 'label_name': name_label_list})
    df1.to_csv(ID_TO_LABEL_NAME_FILE)
    return 0


def build_df_for_purity(ID_TO_LABEL_FILE, ID_TO_LABEL_NAME_FILE, pred):
    df = pd.read_csv(ID_TO_LABEL_FILE)
    A = pd.concat([df, pd.Series(pred, name="cluster")], axis=1)
    df = A[['label_cat', 'cluster']]

    df1 = pd.read_csv(ID_TO_LABEL_NAME_FILE)
    B = pd.concat([df1, pd.Series(pred, name="cluster")], axis=1)
    df1 = B[['label_name', 'cluster']]
    df1.to_csv(ID_TO_LABEL_NAME_FILE)
    return df


# input is a DataFrame with two columns "label_cat" & " "cluster"
def purity(df):
    sum = 0
    for c in df['cluster'].unique():
        sub = df[df['cluster'] == c]
        sum += max(sub['label_cat'].value_counts())

    purity_score = round(sum / len(df), 4)
    print(purity_score)
    return purity_score


def main():
    cluster(**tuning_para)


if __name__ == "__main__":
    main()
    #generate_id_to_label()
