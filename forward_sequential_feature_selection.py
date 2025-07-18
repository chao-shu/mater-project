from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
from tqdm import tqdm
from data_loader import *
import copy

def save_feature(selected_feature, X,label,path):
    X_sub = X[:, selected_feature]

    features_idx = 0
    video_names = sorted(os.listdir(path))
    save_path = './features/combined_feature_with_reduced_pose_mlp_after_selection'  # 新的路径

    os.makedirs(save_path, exist_ok=True)

    for video_name in video_names:
        video_feature_path = os.path.join(path, video_name)
        with open(video_feature_path, 'rb') as f:
            combined_feature = pickle.load(f)

        frame_ids = sorted(combined_feature.keys())
        
        reduced_feature_dict = {}
        for fid in frame_ids:
            reduced_feature_dict[fid] = X_sub[features_idx]
            reduced_feature_dict[fid.split("_")[0]+"_label"] = label[features_idx]
            features_idx += 1

        # 保存降维后的特征
        save_file = os.path.join(save_path, video_name)
        with open(save_file, 'wb') as f:
            pickle.dump(reduced_feature_dict, f)

    print("All PCA-reduced features saved to:", save_path)

def remap_labels(cluster_labels, true_labels, k):
    cluster_to_true = {}
    for cluster_id in range(k):
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) == 0:
            most_common = -1
        else:
            true_for_cluster = [true_labels[i] for i in indices]
            most_common = Counter(true_for_cluster).most_common(1)[0][0]
        cluster_to_true[cluster_id] = most_common
    mapped_pred_labels = np.array([cluster_to_true[cl] for cl in cluster_labels])
    return mapped_pred_labels

def forward_select_accuracy(X, true_labels, K=61, max_features=20):
    """
    Forward feature selection using clustering accuracy as evaluation.

    Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        true_labels (np.ndarray): Ground truth labels
        K (int): Number of clusters for KMeans
        max_features (int): Maximum number of features to select

    Returns:
        selected_features (list): Indices of selected features
        accuracies (list): Accuracy score at each step
    """
    n_samples, n_features = X.shape
    all_features = list(range(n_features))
    selected_features = []
    accuracies = []
    best_selected_feature = None
    best_accuracy = -1

    for step in tqdm(range(max_features), desc="Selecting features"):
        best_acc = -1
        best_feature = None

        for f in all_features:
            if f in selected_features:
                continue
            current_features = selected_features + [f]
            X_sub = X[:, current_features]

            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_sub)
            mapped_pred_labels = remap_labels(cluster_labels, true_labels, K)
            try:
                acc = accuracy_score(true_labels, mapped_pred_labels)
            except:
                print("accuracy is not calculated")
                acc = 0

            if acc > best_acc:
                best_acc = acc
                best_feature = f

        if best_feature is not None:
            selected_features.append(best_feature)
            accuracies.append(best_acc)
        else:
            break
        
        if best_acc > best_accuracy:
            best_accuracy = best_acc
            best_selected_feature = copy.deepcopy(selected_features)
    return selected_features, accuracies, best_selected_feature

path = "./features/combined_feature_with_reduced_pose_mlp"
X, true_labels = combined_feature_loader(path)
# X.shape = (n_samples, 167)
selected, accs, best_selected_feature = forward_select_accuracy(X, true_labels, K=61, max_features=167)

print("Selected features:", best_selected_feature)
print("feature dimension after selection:", len(best_selected_feature))
save_feature(best_selected_feature, X, true_labels, path)

import matplotlib.pyplot as plt
plt.plot(accs, marker='o')
plt.xlabel("Number of Features Selected")
plt.ylabel("Accuracy")
plt.title("Forward Feature Selection using KMeans (K=61)")
plt.grid(True)
plt.show()
