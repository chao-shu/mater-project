import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import csv
from matplotlib.colors import ListedColormap
from collections import Counter
from data_loader import *


def plot_temporal_cluster_feature_tsne(path):

    features, label = temporal_cluster_faeture_loader(path)
    print(len(features),len(label))

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=20, random_state=42)

    features_2d = tsne.fit_transform(features)  # (N, 2)

    # Create a discrete colormap for 3 classes
    cmap = ListedColormap(['red', 'green', 'blue'])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=label, cmap=cmap, s=10)
    
    # Create a custom colorbar with correct ticks
    cbar = plt.colorbar(scatter, ticks=[1, 2, 3])
    cbar.set_label('Emotion Label')
    cbar.set_ticklabels(['1', '2', '3'])

    plt.title('t-SNE of Temporal cluster Features (colored by emotion)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

def plot_temporal_feature_tsne(path):
    features, label = temporal_feature_loader(path)
    # print(len(features),len(label))

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)

    features_2d = tsne.fit_transform(features)  # (N, 2)

    # Create a discrete colormap for 3 classes
    cmap = ListedColormap(['red', 'green', 'blue'])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=label, cmap=cmap, s=10)
    
    # Create a custom colorbar with correct ticks
    cbar = plt.colorbar(scatter, ticks=[1, 2, 3])
    cbar.set_label('Emotion Label')
    cbar.set_ticklabels(['1', '2', '3'])

    plt.title('t-SNE of Reduced Temporal Features (colored by emotion)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

def plot_spatial_feature_tsne(path):
    features, label = spatial_feature_loader(path, "mid")
    label_counts = Counter(label)
    print(label_counts)
    # print(len(features),len(label))

    # # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)

    features_2d = tsne.fit_transform(features)  # (N, 2)

    # Create a discrete colormap for 3 classes
    cmap = ListedColormap(['red', 'green', 'blue'])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=label, cmap=cmap, s=10)
    
    # Create a custom colorbar with correct ticks
    cbar = plt.colorbar(scatter, ticks=[1, 2, 3])
    cbar.set_label('Emotion Label')
    cbar.set_ticklabels(['1', '2', '3'])

    plt.title('t-SNE of Reduced Spatial Features (colored by emotion)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True) 
    plt.show()




def plot_pose_feature_tsne(path):

    feature, label = pose_feature_loader(path)
    label_counts = Counter(label)
    print(label_counts)   

    # # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)

    features_2d = tsne.fit_transform(feature)  # (N, 2)

    # Create a discrete colormap for 3 classes
    cmap = ListedColormap(['red', 'green', 'blue'])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=label, cmap=cmap, s=10)
    
    # Create a custom colorbar with correct ticks
    cbar = plt.colorbar(scatter, ticks=[1, 2, 3])
    cbar.set_label('Emotion Label')
    cbar.set_ticklabels(['1', '2', '3'])

    plt.title('t-SNE of pose Features (colored by emotion)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True) 
    plt.show()

def main():
    label_path = "./data/label/denseframelabel.csv"
    spatial_feature_path =  "./reduced_spatial_feature"
    temporal_feature_path = "./reduced_temporal_feature"
    temporal_cluster_feature_path = "./Temporal_feature_cluster_with_alignment"
    pose_feature_path = "./dataset/facial_data"

    # plot_temporal_feature_tsne(temporal_feature_path)
    # plot_temporal_cluster_feature_tsne(temporal_cluster_feature_path)
    plot_spatial_feature_tsne(spatial_feature_path)
    # plot_pose_feature_tsne(pose_feature_path)
if __name__ == "__main__":
    main()