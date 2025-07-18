import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import csv
from data_loader import *
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score
from collections import Counter

import seaborn as sns

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.tight_layout()
    plt.show()



def plot_cluster_metrics(k_range, inertias, silhouettes, purities, rand_indices, nmis, jaccards):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows, 3 columns

    axs[0, 0].plot(k_range, inertias, marker='o', color='blue')
    axs[0, 0].set_title('Elbow Method (Inertia)')
    axs[0, 0].set_xlabel("Number of Clusters (K)")
    axs[0, 0].set_ylabel("Inertia")
    axs[0, 0].grid(True)

    axs[0, 1].plot(k_range, silhouettes, marker='o', color='black')
    axs[0, 1].set_title("Silhouette Score")
    axs[0, 1].set_xlabel("Number of Clusters (K)")
    axs[0, 1].set_ylabel("Silhouette")
    axs[0, 1].grid(True)

    axs[0, 2].plot(k_range, purities, marker='o', color='green')
    axs[0, 2].set_title("Purity")
    axs[0, 2].set_xlabel("Number of Clusters (K)")
    axs[0, 2].set_ylabel("Purity")
    axs[0, 2].grid(True)

    axs[1, 0].plot(k_range, rand_indices, marker='o', color='orange')
    axs[1, 0].set_title("Adjusted Rand Index")
    axs[1, 0].set_xlabel("Number of Clusters (K)")
    axs[1, 0].set_ylabel("Rand Index")
    axs[1, 0].grid(True)

    axs[1, 1].plot(k_range, nmis, marker='o', color='purple')
    axs[1, 1].set_title("Normalized Mutual Info (NMI)")
    axs[1, 1].set_xlabel("Number of Clusters (K)")
    axs[1, 1].set_ylabel("NMI")
    axs[1, 1].grid(True)

    axs[1, 2].plot(k_range, jaccards, marker='o', color='red')
    axs[1, 2].set_title("Jaccard Score")
    axs[1, 2].set_xlabel("Number of Clusters (K)")
    axs[1, 2].set_ylabel("Jaccard")
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()



def purity_score(y_true, y_pred):
    # Compute contingency matrix (confusion matrix)
    from sklearn.metrics import confusion_matrix
    import numpy as np
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

from sklearn.metrics import accuracy_score, confusion_matrix

def Kmean_train_with_labels(k_range, features, true_labels):
    inertias, silhouettes = [], []
    purities, rand_indices, nmis, jaccards = [], [], [], []
    accuracies = []
    conf_matrices = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        inertias.append(kmeans.inertia_)
        label_counts = Counter(cluster_labels)
        print(f"Cluster counts: {label_counts}")

        if len(features) > k:
            silhouettes.append(silhouette_score(features, cluster_labels))
        else:
            silhouettes.append(None)

        # ====== 重新映射预测标签 ======
        cluster_to_true = {}
        for cluster_id in range(k):
            indices = np.where(cluster_labels == cluster_id)[0]
            if len(indices) == 0:
                most_common = -1  # 或者继续跳过这个 cluster
            else:
                true_for_cluster = [true_labels[i] for i in indices]
                most_common = Counter(true_for_cluster).most_common(1)[0][0]
            cluster_to_true[cluster_id] = most_common

        mapped_pred_labels = np.array([cluster_to_true[cl] for cl in cluster_labels])

        # ===== 打印每个 cluster 的真实标签分布 =====
        print("Label distribution in each cluster:")
        for cluster_id in range(k):
            indices = np.where(cluster_labels == cluster_id)[0]
            true_labels_in_cluster = [true_labels[i] for i in indices]
            dist = Counter(true_labels_in_cluster)
            print(f"  Cluster {cluster_id}: {dict(dist)}")

        # ====== 评价指标 ======
        purities.append(purity_score(true_labels, cluster_labels))
        rand_indices.append(adjusted_rand_score(true_labels, cluster_labels))
        nmis.append(normalized_mutual_info_score(true_labels, cluster_labels))
        
        try:
            jaccard = jaccard_score(true_labels, mapped_pred_labels, average='macro')
        except:
            jaccard = None
        jaccards.append(jaccard)

        acc = accuracy_score(true_labels, mapped_pred_labels)
        accuracies.append(acc)

        conf_mat = confusion_matrix(true_labels, mapped_pred_labels)
        conf_matrices.append(conf_mat)

        print(f"K={k}: Inertia={inertias[-1]:.2f}, Silhouette={silhouettes[-1]}, "
              f"Purity={purities[-1]:.4f}, Rand={rand_indices[-1]:.4f}, "
              f"Accuracy={acc:.4f}, Jaccard={jaccard}")

    return inertias, silhouettes, purities, rand_indices, nmis, jaccards, accuracies, conf_matrices


def main():
    dataset=9

    if dataset == 1:
        path = "./features/Temporal_feature"
        features, labels = temporal_feature_loader(path)# elbol 7
    elif dataset == 2:
        path = "./features/S_former_feature"
        features, labels = spatial_feature_loader(path, "end")
    elif dataset==3:
        path = "./features/S_former_feature"
        features, labels = spatial_feature_loader(path, "mid")
    elif dataset ==4:
        path = "./dataset/facial_data"
        features, labels = pose_feature_loader(path)

    if dataset == 5:
        path = "./features/reduced_temporal_feature"
        features, labels = temporal_feature_loader(path)# elbol 7
    elif dataset == 6:
        path = "./features/reduced_spatial_feature"
        features, labels = spatial_feature_loader(path, "end")
    elif dataset==7:
        path = "./features/reduced_spatial_feature"
        features, labels = spatial_feature_loader(path, "mid")
    elif dataset ==8:
        path = "./features/reduced_pose_feature"
        features, labels = reduced_pose_feature_loader(path)
    elif dataset ==9:
        path = "./features/combined_feature_with_reduced_pose_mlp"
        features, labels = combined_feature_loader(path)


    label_counts = Counter(labels)
    print(f"true label distribution: {label_counts}")

    print(len(features))
    print("data loading successful")
    from sklearn.preprocessing import StandardScaler
    features = StandardScaler().fit_transform(features)

    k_range=range(3, 101)
    inertias, silhouettes, purities, rand_indices, nmis, jaccards , accuracies, conf_matrices= Kmean_train_with_labels(k_range, features, labels)

    # plot_cluster_metrics(k_range, inertias, silhouettes, purities, rand_indices, nmis, jaccards)

    # plot_confusion_matrix(conf_matrices[0], sorted(set(labels)), title=f"K={k_range[0]} Confusion Matrix")

    for i, k in enumerate(k_range):
        print(f"\nConfusion Matrix for K={k}:")
        print(conf_matrices[i])
        print(f"Accuracy: {accuracies[i]:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, accuracies, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Clusters')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()