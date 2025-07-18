import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from collections import Counter
from sklearn.decomposition import PCA
from data_loader import *

def pca_proportion_draw(pca):

    # Show PCA explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("Explained variance ratio of each component:")
    for i, (evr, cum) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"PC{i+1}: {evr:.4f} (cumulative: {cum:.4f})")

    # Optional: Plot the cumulative variance
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def read_dense_frame_labels(fname):

    label_list = []
    with open("./data/label/denseframelabel.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == fname:
                label_list.append(row)
    return label_list
def normalize(arr, method='zscore', stats=None):
    arr = np.array(arr)
    if method == 'zscore':
        mean = stats['mean']
        std = stats['std']
        return (arr - mean) / (std)
    elif method == 'minmax':
        min_val = stats['min']
        max_val = stats['max']
        return (arr - min_val) / (max_val - min_val + 1e-8)
    else:
        return arr  # No normalization

def compute_zscore_stats(path):
    aus_list = []
    emo_list = []
    head_pose_list = []
    gaze_list = []
    landmarks_list = []

    video_names = sorted(os.listdir(path))
    for video_name in video_names:
        video_path = os.path.join(path, video_name)
        frame_names = sorted(os.listdir(video_path))
        for frame_name in frame_names:
            frame_path = os.path.join(video_path, frame_name)
            if frame_path in non_pose:
                continue

            with open(frame_path, 'rb') as f:
                feat = pickle.load(f)
            aus_list.append(feat['aus'])
            emo_list.append(feat['emo'])
            head_pose_list.append([
                feat['head_pose']['pitch'],
                feat['head_pose']['yaw'],
                feat['head_pose']['roll']
            ])
            gaze_list.append([
                feat['gaze']['pitch'],
                feat['gaze']['yaw'],
                feat['gaze']['vector'][0]
            ])
            landmarks_list.append(feat['facial_landmarks'].flatten())

    stats = {
        'aus': {'mean': np.mean(aus_list, axis=0), 'std': np.std(aus_list, axis=0)},
        'emo': {'mean': np.mean(emo_list, axis=0), 'std': np.std(emo_list, axis=0)},
        'head_pose': {'mean': np.mean(head_pose_list, axis=0), 'std': np.std(head_pose_list, axis=0)},
        'gaze': {'mean': np.mean(gaze_list, axis=0), 'std': np.std(gaze_list, axis=0)},
        'landmarks': {'mean': np.mean(landmarks_list, axis=0), 'std': np.std(landmarks_list, axis=0)},
    }
    print(len(stats["aus"]["mean"]), len(stats["emo"]["mean"]),len(stats["head_pose"]["mean"]),len(stats["gaze"]["mean"]),len(stats["landmarks"]["mean"]))
    return stats

def flatten_features(feat, normalization_stats=None):
    parts = []
    method='zscore'

    # 1. AU + emotion
    aus = np.array(feat['aus'])
    emo = np.array(feat['emo'])
    if normalization_stats:
        aus = normalize(aus, method, normalization_stats['aus'])
        emo = normalize(emo, method, normalization_stats['emo'])
    parts.append(aus)
    parts.append(emo)
    
    # 2. Head-pose
    hp = np.array([feat['head_pose']['pitch'], feat['head_pose']['yaw'], feat['head_pose']['roll']])
    if normalization_stats:
        hp = normalize(hp, method, normalization_stats['head_pose'])
    parts.append(hp)
    
    # 3. Gaze
    gz = np.array([feat['gaze']['pitch'], feat['gaze']['yaw'], feat['gaze']['vector'][0]])
    if normalization_stats:
        gz = normalize(gz, method, normalization_stats['gaze'])
    parts.append(gz)
    
    # 4. Landmarks
    lm = feat['facial_landmarks'].flatten()
    if normalization_stats:
        lm = normalize(lm, method, normalization_stats['landmarks'])
    parts.append(lm)
    
    return np.concatenate(parts)


def z_normalization(feature):
    mean = np.mean(feature,axis=0)
    std = np.std(feature, axis=0)
    std[std == 0] = 1e-8
    return (feature-mean)/std

def reduce_pose_mlp_feature(path):

    features, label = reduced_pose_feature_loader(path)
    features=z_normalization(features)

    print(len(features),len(label))
    print("original dimension:", features.shape[1])
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features)
    print("saved dimension",features_reduced.shape[1])

    min_val = features.min()
    max_val = features.max()
    mean_val = features.mean()
    std_val = features.std()
    print(min_val, max_val, mean_val, std_val)

    pca_proportion_draw(pca)
    
    # save reduced features
    features_idx = 0
    video_names = sorted(os.listdir(path))
    save_path = './features/normalized_reduced_pose_mlp_feature'  # 新的路径

    os.makedirs(save_path, exist_ok=True)

    for video_name in video_names:
        video_feature_path = os.path.join(path, video_name)
        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        frame_ids = sorted(temporal_features.keys())
        
        reduced_feature_dict = {}
        for fid in frame_ids:
            if fid.split("_")[1]== "face":

                reduced_feature_dict[fid] = features_reduced[features_idx]
                reduced_feature_dict[fid.split("_")[0]+"_label"]=label[features_idx]
                features_idx += 1

        # 保存降维后的特征
        save_file = os.path.join(save_path, video_name)
        with open(save_file, 'wb') as f:
            pickle.dump(reduced_feature_dict, f)

    print("All PCA-reduced features saved to:", save_path)

def reduce_temporal_feature(path):

    features, label = temporal_feature_loader(path)
    features=z_normalization(features)
    print(len(features),len(label))
    print("original dimension:", features.shape[1])
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features)
    print("saved dimension",features_reduced.shape[1])

    min_val = features.min()
    max_val = features.max()
    mean_val = features.mean()
    std_val = features.std()
    print(min_val, max_val, mean_val, std_val)

    pca_proportion_draw(pca)
    
    # save reduced features
    features_idx = 0
    video_names = sorted(os.listdir(path))
    save_path = './normalized_reduced_temporal_feature'  # 新的路径

    os.makedirs(save_path, exist_ok=True)

    for video_name in video_names:
        video_feature_path = os.path.join(path, video_name)
        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        frame_ids = sorted(temporal_features.keys())
        
        reduced_feature_dict = {}
        for fid in frame_ids:
            reduced_feature_dict[fid] = features_reduced[features_idx]
            features_idx += 1

        # 保存降维后的特征
        save_file = os.path.join(save_path, video_name)
        with open(save_file, 'wb') as f:
            pickle.dump(reduced_feature_dict, f)

    print("All PCA-reduced features saved to:", save_path)

def reduce_spatial_feature(path):

    features_mid, label = spatial_feature_loader(path, "mid")
    features_end, _ = spatial_feature_loader(path, "end")

    features_mid = z_normalization(features_mid)
    features_end = z_normalization(features_end)

    print("number of features in end, mid, label", len(features_end),len(features_mid),len(label))

    print("original dimension of mid:", features_mid.shape[1])
    mid_pca = PCA(n_components=0.95)
    features_mid_reduced = mid_pca.fit_transform(features_mid)
    print("saved dimension of mid",features_mid_reduced.shape[1])

    print("original dimension of end:", features_end.shape[1])
    end_pca = PCA(n_components=0.95)
    features_end_reduced = end_pca.fit_transform(features_end)
    print("saved dimension of end",features_end_reduced.shape[1])
    
    print("features_mid distribution------")
    min_val = features_mid.min()
    max_val = features_mid.max()
    mean_val = features_mid.mean()
    std_val = features_mid.std()
    print(min_val, max_val, mean_val, std_val)

    print("features_end distribution------")
    min_val = features_end.min()
    max_val = features_end.max()
    mean_val = features_end.mean()
    std_val = features_end.std()
    print(min_val, max_val, mean_val, std_val)

    pca_proportion_draw(end_pca)
    pca_proportion_draw(mid_pca)

    # save reduced features
    save_path = 'normalized_reduced_spatial_feature'  # final output directory
    os.makedirs(save_path, exist_ok=True)

    idx_mid = 0
    idx_end = 0

    video_names = sorted(os.listdir(path))

    for video_name in video_names:
        video_feature_path = os.path.join(path, video_name)
        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        frame_ids = sorted(temporal_features.keys())
        reduced_feature_dict = {}

        for fid in frame_ids:
            parts = fid.split("_")
            frame_type = parts[2]

            if frame_type == "mid":
                reduced_feature_dict[fid] = features_mid_reduced[idx_mid]
                idx_mid += 1
            elif frame_type == "end":
                reduced_feature_dict[fid] = features_end_reduced[idx_end]
                idx_end += 1

        # Save into one single .pkl file per video (same format as before)
        save_file = os.path.join(save_path, video_name)
        with open(save_file, 'wb') as f:
            pickle.dump(reduced_feature_dict, f)

    print("All reduced features saved in the same structure as before at:", save_path)

non_pose = [
    './dataset/facial_data/1657025477939515/0368.pkl',
    './dataset/facial_data/16571735778363602/0417.pkl',
    './dataset/facial_data/16571735778363602/0537.pkl',
    './dataset/facial_data/16571735778363602/0538.pkl',
    './dataset/facial_data/16571735778363602/0539.pkl',
    './dataset/facial_data/16571735778363602/0540.pkl'

]
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict, Counter

def reduce_pose_feature(path):
    all_features = []
    all_labels = []
    video_frame_map = []  # (video_name, frame_name)

    video_names = sorted(os.listdir(path))
    feature_stats = compute_zscore_stats(path)
    for video_name in video_names:
        video_path = os.path.join(path, video_name)
        frame_names = sorted(os.listdir(video_path))
        for frame_name in frame_names:
            frame_path = os.path.join(video_path, frame_name)
            if frame_path in non_pose:
                continue

            with open(frame_path, 'rb') as f:
                frame_data = pickle.load(f)

            vec = flatten_features(frame_data,feature_stats)
            if np.isnan(vec).any():
                print(f"⚠️ NaN detected in {frame_path}")
                continue

            all_features.append(vec)
            all_labels.append(frame_data["frame_label"])
            video_frame_map.append((video_name, frame_name))  # For later mapping

    all_features = np.array(all_features)
    all_labels = np.array(all_labels).astype(int)

    print("Label distribution:", Counter(all_labels))
    print("Original dimension:", all_features.shape[1])

    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(all_features)


    print("Reduced dimension:", reduced_features.shape[1])
    min_val = all_features.min()
    max_val = all_features.max()
    mean_val = all_features.mean()
    std_val = all_features.std()
    print(min_val, max_val, mean_val, std_val)

    # pca_proportion_draw(pca)


    # Group features per video
    per_video_data = defaultdict(dict)
    for i, (video_name, frame_name) in enumerate(video_frame_map):
        frame_id = os.path.splitext(frame_name)[0]  # e.g. '00001'
        per_video_data[video_name][f"{frame_id}_face"] = reduced_features[i]
        per_video_data[video_name][f"{frame_id}_label"] = all_labels[i]

    # Save each video_name.pkl
    save_path = "./features/normalized_reduced_pose_feature"
    os.makedirs(save_path, exist_ok=True)
    for video_name, frame_dict in per_video_data.items():
        save_file = os.path.join(save_path, f"{video_name}.pkl")
        with open(save_file, 'wb') as f:
            pickle.dump(frame_dict, f)

    print(f"✅ Saved all reduced features to: {save_path}")

def main():
    label_path = "./data/label/denseframelabel.csv"
    spatial_feature_path =  "./S_former_feature"
    temporal_feature_path = "./Temporal_feature"
    pose_feature_path = "./dataset/facial_data"

    # reduce_temporal_feature(temporal_feature_path)
    # reduce_spatial_feature(spatial_feature_path)
    # reduce_pose_feature(pose_feature_path)
    reduce_pose_mlp_feature("./features/pose_mlp_feature")
if __name__ == "__main__":
    main()