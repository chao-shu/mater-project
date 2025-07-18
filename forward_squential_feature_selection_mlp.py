import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy

from data_loader import *  # Must return (X, labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_and_evaluate(X_train, y_train, X_val, y_val, epochs=150):
    input_dim = X_train.shape[1]
    model = MLP(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Shift labels to [0, 1, 2]
    y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long).to(device)
    y_val_tensor = torch.tensor(y_val - 1, dtype=torch.long).to(device)

    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_val_tensor).float().mean().item()

    return acc

def feature_loader(path):
    features = []
    label = []
    test_features = []
    test_label = []
    
    video_names = sorted(os.listdir(path))
    
    for video_name in video_names:
        video_feature_path = os.path.join(path, video_name)
        video_id = video_name.split('.')[0]

        with open(video_feature_path, 'rb') as f:
            temporal_features = pickle.load(f)

        label_list = read_dense_frame_labels(video_id)  
        topic_list = read_dense_topic_labels(video_id)
        # Sorted list of frame keys
        frame_ids = sorted(temporal_features.keys(), key=lambda x: int(x.split("_")[0]))

        # Detect blocks with same topic
        current_topic = None
        current_block = []
        for k in frame_ids:
            frame = int(k.split("_")[0])
            topic = topic_list[frame][2]

            if topic != current_topic:
                # Process previous block
                if current_block:
                    split_block(current_block, label_list, temporal_features, features, label, test_features, test_label)
                # Start new block
                current_block = [(k, frame)]
                current_topic = topic
            else:
                current_block.append((k, frame))

        # Don't forget the last block
        if current_block:
            split_block(current_block, label_list, temporal_features, features, label, test_features, test_label)
    # Convert to numpy arrays
    features = np.array(features)
    test_features = np.array(test_features)
    label = np.array(label).astype(int)
    test_label = np.array(test_label).astype(int)

    return features, label, test_features, test_label


def split_block(block, label_list, temporal_features, features, label, test_features, test_label):
    """Split a topic block into train/test halves."""
    n = len(block)
    split = n // 2
    for i, (k, frame) in enumerate(block):
        feat = temporal_features[k]
        lbl = label_list[frame][2]
        if i < split:
            features.append(feat)
            label.append(lbl)
        else:
            test_features.append(feat)
            test_label.append(lbl)

def forward_select_pytorch(path, max_features=20):
    n_samples, n_features = X.shape
    all_features = list(range(n_features))
    selected_features = []
    accuracies = []
    best_accuracy = -1
    best_selected_feature = None

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, stratify=y, random_state=42
    # )
    X_train, y_train, X_val, y_val = feature_loader(path)

    for step in tqdm(range(max_features), desc="Selecting features"):
        best_acc = -1
        best_feature = None

        for f in all_features:
            if f in selected_features:
                continue

            current_features = selected_features + [f]
            X_train_sub = X_train[:, current_features]
            X_val_sub = X_val[:, current_features]

            acc = train_and_evaluate(X_train_sub, y_train, X_val_sub, y_val)

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

def save_feature(selected_feature, X, label, path):
    X_sub = X[:, selected_feature]

    features_idx = 0
    video_names = sorted(os.listdir(path))
    save_path = './features/combined_feature_with_reduced_pose_mlp_after_selection_with_mlp_half_half'

    os.makedirs(save_path, exist_ok=True)

    for video_name in video_names:
        video_feature_path = os.path.join(path, video_name)
        with open(video_feature_path, 'rb') as f:
            combined_feature = pickle.load(f)

        frame_ids = sorted(combined_feature.keys())
        
        reduced_feature_dict = {}
        for fid in frame_ids:
            reduced_feature_dict[fid] = X_sub[features_idx]
            reduced_feature_dict[fid.split("_")[0] + "_label"] = label[features_idx]
            features_idx += 1

        save_file = os.path.join(save_path, video_name)
        with open(save_file, 'wb') as f:
            pickle.dump(reduced_feature_dict, f)

    print("All PyTorch-selected features saved to:", save_path)

# --------------------------- Main Logic ---------------------------

if __name__ == "__main__":
    path = "./features/combined_feature_with_reduced_pose_mlp"
    X, y = combined_feature_loader(path)  # (n_samples, n_features), labels

    print("Starting PyTorch-based forward feature selection...")
    selected, accs, best_selected_feature = forward_select_pytorch(path, max_features=167)

    print("\nBest selected features:", best_selected_feature)
    print("Feature dimension after selection:", len(best_selected_feature))

    save_feature(best_selected_feature, X, y, path)

    # Plot
    plt.plot(accs, marker='o')
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Accuracy")
    plt.title("Forward Feature Selection using PyTorch MLP (GPU)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
