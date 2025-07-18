import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import csv
def read_dense_frame_labels(fname):

    label_list = []
    with open("./data/label/denseframelabel.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == fname:
                label_list.append(row)
    return label_list

def read_dense_topic_labels(fname):
    label_list = []
    with open("./data/label/densetopiclabel.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == fname:
                label_list.append(row)
    return label_list

def temporal_feature_loader(path):
    features=[]
    label = []
    video_names = sorted(os.listdir(path))
    for video_name in video_names:
        video_feature_path = os.path.join(path,video_name)
        label_list = read_dense_frame_labels(video_name.split('.')[0])
        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        # Extract features and IDs
        frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
        for k in frame_ids:

            features.append(temporal_features[k])
            frame = int(k.split("_")[1])
            label.append(label_list[frame][2])
    features = np.array(features)
    label = np.array(label).astype(int) 
    return features, label

def temporal_cluster_faeture_loader(path):
    features=[]
    label = []
    video_names = sorted(os.listdir(path))
    for video_name in video_names:
        video_feature_path = os.path.join(path,video_name)
        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        # Extract features and IDs
        frame_ids = sorted(set(k for k in temporal_features if 'temporal_output' in k))
        # print("frame size and label size",video_name,len(frame_ids), len(label_list))
        for k in frame_ids:

            features.append(temporal_features[k])
            frame = k.split("_")[1]
            label.append(temporal_features["label_"+frame])
    features = np.array(features)
    label = np.array(label).astype(int) 
    return features, label
  
def spatial_feature_loader(path, feature_type):
    features=[]
    label = []
    video_names = sorted(os.listdir(path))
    for video_name in video_names:
        video_feature_path = os.path.join(path,video_name)
        label_list = read_dense_frame_labels(video_name.split('.')[0])
        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        # Extract features and IDs
        frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
        # print("frame size and label size",video_name,len(frame_ids), len(label_list))
        for k in frame_ids:
            if k.split("_")[2]==feature_type:
                features.append(temporal_features[k])
                frame = int(k.split("_")[1])
                label.append(label_list[frame][2])
    features = np.array(features)
    label = np.array(label).astype(int) 
    return features, label

def flatten_features(feat):
    parts = []
    

    
    # 1. AU + emotion
    parts.append(feat['aus'])   # ~20 dim
    parts.append(feat['emo'])   # 7 dim
    
    # 2. Head-pose
    hp = feat['head_pose']
    parts.append(np.array([hp['pitch'], hp['yaw'], hp['roll']]))
    
    # 3. Gaze
    gz = feat['gaze']
    parts.append(np.array([gz['pitch'], gz['yaw'], gz['vector'][0]]))
    
    # 4. Landmarks
    lm = feat['facial_landmarks']  # shape (N, 2)
    parts.append(lm.flatten())
    
    # Concatenate
    
    return np.concatenate(parts) 

non_pose = [
    './dataset/facial_data/1657025477939515/0368.pkl',
    './dataset/facial_data/16571735778363602/0417.pkl',
    './dataset/facial_data/16571735778363602/0537.pkl',
    './dataset/facial_data/16571735778363602/0538.pkl',
    './dataset/facial_data/16571735778363602/0539.pkl',
    './dataset/facial_data/16571735778363602/0540.pkl'

]
def pose_feature_loader(path):
    feature = []
    label = []

    video_names = sorted(os.listdir(path))
    for video_name in video_names:
        video_feature_path = os.path.join(path,video_name)
        frame_names= sorted(os.listdir(video_feature_path))
        for image_feature in frame_names:
            image_feature_path = os.path.join(video_feature_path,image_feature)
            if image_feature_path in non_pose:
                continue
            with open(image_feature_path, 'rb') as f:
                temporal_features = pickle.load(f)

            f=flatten_features(temporal_features)
            # print("feature length"+str(len(f)))
            if np.isnan(f).any():
                print(f"NaN detected in: {image_feature_path}")
            feature.append(f)
            label.append(temporal_features["frame_label"])

    feature = np.array(feature)
    label = np.array(label).astype(int) 
    return feature, label

def reduced_pose_feature_loader(path):
    features=[]
    label = []
    video_names = sorted(os.listdir(path))
    for video_name in video_names:
        video_feature_path = os.path.join(path,video_name)
        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        # Extract features and IDs
        frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
        # print("frame size and label size",video_name,len(frame_ids), len(label_list))
        for k in frame_ids:
            if k.split("_")[1] =="face":
                features.append(temporal_features[k])
                label.append(temporal_features[k.split("_")[0]+'_label'])
    features = np.array(features)
    label = np.array(label).astype(int) 
    return features, label

def combined_feature_loader(path):
    features=[]
    label = []
    video_names = sorted(os.listdir(path))
    for video_name in video_names:
        video_feature_path = os.path.join(path,video_name)
        label_list = read_dense_frame_labels(video_name.split('.')[0])
        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        # Extract features and IDs
        frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
        for k in frame_ids:

            features.append(temporal_features[k])
            frame = int(k.split("_")[0])
            label.append(label_list[frame][2])
    features = np.array(features)
    label = np.array(label).astype(int) 
    return features, label
