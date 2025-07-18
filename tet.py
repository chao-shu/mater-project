import pickle
import os
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import csv
from matplotlib.colors import ListedColormap
from collections import Counter# # Load the .pkl file
from data_loader import *




def combine_feature():
    label = []
    path='./features/normalized_reduced_temporal_feature'
    video_names = sorted(os.listdir(path))
    for video_name in video_names:

        temporal_video_feature_path = os.path.join(path,video_name)
        spatial_video_feature_path= os.path.join('./features/normalized_reduced_spatial_feature',video_name)
        pose_video_feature_path= os.path.join("./features/normalized_reduced_pose_mlp_feature",video_name)

        label_list = read_dense_frame_labels(video_name.split('.')[0])
        with open(temporal_video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)
        with open(spatial_video_feature_path, 'rb') as f:
            spatial_features = pickle.load(f)
        with open(pose_video_feature_path, 'rb') as f:
            pose_features = pickle.load(f)

        # Extract features and IDs
        temporal_frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
        spatial_frame_ids = sorted(spatial_features.keys())  # Keep them sorted by frame
        pose_frame_ids = sorted(pose_features.keys())  # Keep them sorted by frame

        features={}
        for k in temporal_frame_ids:
            frame = k.split("_")[1]
            temp_feature = []
            if frame[-4:]+"_face" in pose_frame_ids and "face_"+frame+"_mid_output" in spatial_frame_ids:
                temp_feature.append(temporal_features[k])
                temp_feature.append(spatial_features["face_"+frame+"_mid_output"])
                temp_feature.append(spatial_features["face_"+frame+"_end_output"])
                # temp_feature.append(pose_features[frame[-4:]+"_face"])


                features[frame+"_face"] = np.array(np.concatenate(temp_feature))
                label.append(label_list[int(frame)][2])
                if (int(label_list[int(frame)][2]) != int(pose_features[frame[-4:]+"_label"])):
                    print("label not match")
                    print(label_list[int(frame)][2] , pose_features[frame[-4:]+"_label"])
        # 保存降维后的特征
        save_file = os.path.join("./features/combined_zhang_feature", video_name)
        with open(save_file, 'wb') as f:
            pickle.dump(features, f)
            
    label = np.array(label).astype(int) 


# combine_feature()








# features, _ = pose_feature_loader("./dataset/facial_data")
# # Across the whole array
# min_val = features.min()
# max_val = features.max()
# mean_val = features.mean()
# std_val = features.std()

# # Per feature (i.e., column-wise)
# min_per_feature = features.min(axis=0)
# max_per_feature = features.max(axis=0)
# mean_per_feature = features.mean(axis=0)
# std_per_feature = features.std(axis=0)
# print(min_val, max_val, mean_val, std_val)


# def flatten_features(feat):
#     parts = []
    

    
#     # 1. AU + emotion
#     parts.append(feat['aus'])   # ~20 dim
#     parts.append(feat['emo'])   # 7 dim
    
#     # 2. Head-pose
#     hp = feat['head_pose']
#     parts.append(np.array([hp['pitch'], hp['yaw'], hp['roll']]))
    
#     # 3. Gaze
#     gz = feat['gaze']
#     parts.append(np.array([gz['pitch'], gz['yaw'], gz['vector'][0]]))
    
#     # 4. Landmarks
#     lm = feat['facial_landmarks']  # shape (N, 2)
#     parts.append(lm.flatten())
    
#     # Concatenate
    
#     return np.concatenate(parts) 

# path='./reduced_pose_feature/16400282145.pkl'
path1= "./features/combined_feature_with_reduced_pose_mlp_after_selection/16400282145.pkl"
with open(path1, 'rb') as f:
    feature_dict = pickle.load(f)
    # feature_dict=flatten_features(feature_dict)
# Print the content of the loaded object (the dictionary)
print(feature_dict)
    


# video_names = sorted(os.listdir("./Temporal_feature"))
# for video_name in video_names:
#     video_feature_path = os.path.join("./Temporal_feature",video_name)
#     with open(video_feature_path, 'rb') as f:
#         temporal_features = pickle.load(f)
#     print(video_name, len(temporal_features))


# video_names = sorted(os.listdir("./dataset/facial_data"))
# for video_name in video_names:
#     video_feature_path = os.path.join("./dataset/facial_data",video_name)
#     frame_names= sorted(os.listdir(video_feature_path))

#     print(video_name, len(frame_names))


# def plot_pose_feature_tsne(path):

#     features=[]
#     label = []
#     video_names = sorted(os.listdir(path))
#     for video_name in video_names:
#         video_feature_path = os.path.join(path,video_name)
#         with open(video_feature_path, 'rb') as f:
#             temporal_features = pickle.load(f)

#         # Extract features and IDs
#         frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
#         # print("frame size and label size",video_name,len(frame_ids), len(label_list))
#         for k in frame_ids:
#             if k.split("_")[1] =="face":
#                 features.append(temporal_features[k])
#                 label.append(temporal_features[k.split("_")[0]+'_label'])
#     features = np.array(features)
#     label = np.array(label).astype(int) 
#     # print(len(features),len(label))

#     # Run t-SNE
#     tsne = TSNE(n_components=2, perplexity=50, random_state=42)

#     features_2d = tsne.fit_transform(features)  # (N, 2)

#     # Create a discrete colormap for 3 classes
#     cmap = ListedColormap(['red', 'green', 'blue'])

#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=label, cmap=cmap, s=10)
    
#     # Create a custom colorbar with correct ticks
#     cbar = plt.colorbar(scatter, ticks=[1, 2, 3])
#     cbar.set_label('Emotion Label')
#     cbar.set_ticklabels(['1', '2', '3'])

#     plt.title('t-SNE of Reduced Temporal Features (colored by emotion)')
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.grid(True)
#     plt.show()
# plot_pose_feature_tsne("reduced_pose_feature")