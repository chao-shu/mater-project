import torch
import torch.nn as nn

# class FeatureMLP(nn.Module):
#     def __init__(self):
#         super(FeatureMLP, self).__init__()
#         self.fc1 = nn.Linear(160,128)
#         self.relu1 = nn.ReLU()

#         self.fc2 = nn.Linear(128, 64)
#         self.relu2 = nn.ReLU()

#         self.fc3 = nn.Linear(64, 3)


#     def forward(self, x):
#         x = self.relu2(self.fc2(self.relu1(self.fc1(x))))
#         logits = self.fc3(x)
#         return logits
    
    
class FeatureMLP(nn.Module):
    def __init__(self):
        super(FeatureMLP, self).__init__()
        self.fc1 = nn.Linear(31,16)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(8, 3)


    def forward(self, x):
        x = self.relu2(self.fc2(self.relu1(self.fc1(x))))
        logits = self.fc3(x)
        return logits
    


import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
from data_loader import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns
from tqdm import tqdm
import numpy as np
from scipy.stats import mode

video_dict = {
            "1658828085789384": 590, #other woman
            "165745616715997": 530, #other man
            "16572216818408213": 650, # south A woman
            "16580176851586106": 556, # south A man
            # "1657019500197831": 656, # south A man
            "1657025477939515": 736, # other man
            "16569562540764549": 657, #east A man
            "16570365844633188": 656, #east A man 
            "16571735778363602": 660, #east A woman
            "16572034690729465": 660, #east A woman
            # "16643616205681827": 629, #east A man
            "16638490487013648": 664, #south A woman
            "16637850973931577": 658, #other woman
            "16637915350525117": 656, #other woman
            "16639504750710998": 658, #east A man
            "16638397654914925": 483, #east A woman
            "16400282145": 577, #other woman
            "163904405748": 576, #south A woman
            "163904657409": 533, #east A man
            "163965258378": 577, #other woman
            "163974758282": 579, #other woman 
            "164002054426": 576, #other man
            "164006412052": 575, #east A woman
            "164007913498": 574, #other man
            "164008493608": 577 #east A woman
            } 

class half_half():
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
                        half_half.split_block(current_block, label_list, temporal_features, features, label, test_features, test_label)
                    # Start new block
                    current_block = [(k, frame)]
                    current_topic = topic
                else:
                    current_block.append((k, frame))

            # Don't forget the last block
            if current_block:
                half_half.split_block(current_block, label_list, temporal_features, features, label, test_features, test_label)
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

    def feature_loader2(path):
        features = []
        label = []
        test_features = []
        test_label = []

        video_names = sorted(os.listdir(path))

        for video_name in video_names:
            video_feature_path = os.path.join(path, video_name)

            with open(video_feature_path, 'rb') as f:
                temporal_features = pickle.load(f)

            # Get all frame IDs (e.g., "0001") that have "face" features
            frame_ids = sorted([
                k.split("_")[0] for k in temporal_features.keys()
                if k.endswith("_face")
            ], key=lambda x: int(x))

            current_topic = None
            current_block = []

            for fid in frame_ids:
                topic = temporal_features[fid + "_label"]

                if topic != current_topic:
                    if current_block:
                        half_half.split_block2(current_block, temporal_features, features, label, test_features, test_label)
                    current_block = [fid]
                    current_topic = topic
                else:
                    current_block.append(fid)

            if current_block:
                half_half.split_block2(current_block, temporal_features, features, label, test_features, test_label)

        # Convert to numpy arrays
        features = np.array(features)
        test_features = np.array(test_features)
        label = np.array(label).astype(int)
        test_label = np.array(test_label).astype(int)

        return features, label, test_features, test_label


    def split_block2(block, temporal_features, features, label, test_features, test_label):
        """Split a topic block into train/test halves based on face features."""
        n = len(block)
        split = n // 2
        for i, fid in enumerate(block):
            face_key = fid + "_face"
            lbl_key = fid + "_label"

            if face_key not in temporal_features or lbl_key not in temporal_features:
                print("missing frame")
                continue  # skip if missing

            feat = temporal_features[face_key]
            lbl = temporal_features[lbl_key]

            if i < split:
                features.append(feat)
                label.append(lbl)
            else:
                test_features.append(feat)
                test_label.append(lbl)

    def half_half_train(path, operation):
        if operation == 3:
            X_train, y_train, X_test, y_test = half_half.feature_loader(path)
        elif operation==4:
            X_train, y_train, X_test, y_test = half_half.feature_loader2(path)
        elif operation ==5:
            X_train, y_train, X_test, y_test = half_half.feature_loader2(path)
        print(len(X_test), len(X_train))
        model = leave_one_out.train(X_train,y_train)
        y_true, y_pred = leave_one_out.test(X_test, y_test, model)
 
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)

        # Per-class accuracy
        for i in range(3):
            correct = np.sum((y_pred == i) & (y_true == i))
            total = np.sum(y_true == i)
            acc = correct / total * 100 if total > 0 else 0
            print(f"Class {i+1} Accuracy: {acc:.2f}%")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
            labels=[0, 1, 2],
            target_names=["Class 1", "Class 2", "Class 3"],
            zero_division=0))


        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 1", "Class 2", "Class 3"], yticklabels=["Class 1", "Class 2", "Class 3"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

def smooth_labels(labels, window=1):

    labels = np.array(labels)
    smoothed = np.empty_like(labels)
    T = len(labels)
    
    for t in range(T):
        start = max(0, t - window)
        end = min(T, t + window + 1)
        neighborhood = labels[start:end]
        smoothed[t] = mode(neighborhood, keepdims=False).mode  # Majority vote
    
    return smoothed       
class leave_one_out():
    def feature_loader(path, test_name):
        features=[]
        label = []
        test_features = []
        test_label = []
        video_names = sorted(os.listdir(path))
        for video_name in video_names:
            video_feature_path = os.path.join(path,video_name)
            label_list = read_dense_frame_labels(video_name.split('.')[0])
            with open(video_feature_path, 'rb') as f:
                temporal_features = pickle.load(f)

            if video_name.split(".")[0] == test_name:
                frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
                for k in frame_ids:

                    test_features.append(temporal_features[k])
                    frame = int(k.split("_")[0])
                    test_label.append(label_list[frame][2])            
            elif video_name.split(".")[0] in video_dict:
                # Extract features and IDs
                frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
                for k in frame_ids:

                    features.append(temporal_features[k])
                    frame = int(k.split("_")[0])
                    label.append(label_list[frame][2])
        features = np.array(features)
        test_features = np.array(test_features)
        label = np.array(label).astype(int) 
        test_label = np.array(test_label).astype(int)
        return features, label, test_features, test_label
        
    def feature_loader2(path, test_name):
        features=[]
        label = []
        test_features = []
        test_label = []

        video_names = sorted(os.listdir(path))
        for video_name in video_names:
            video_feature_path = os.path.join(path,video_name)
            with open(video_feature_path, 'rb') as f:
                temporal_features = pickle.load(f)

            if video_name.split(".")[0] == test_name:
                frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
                # print("frame size and label size",video_name,len(frame_ids), len(label_list))
                for k in frame_ids:
                    if k.split("_")[1] =="face":
                        test_features.append(temporal_features[k])
                        test_label.append(temporal_features[k.split("_")[0]+'_label'])
            elif video_name.split(".")[0] in video_dict:
                # Extract features and IDs
                frame_ids = sorted(temporal_features.keys())  # Keep them sorted by frame
                # print("frame size and label size",video_name,len(frame_ids), len(label_list))
                for k in frame_ids:
                    if k.split("_")[1] =="face":
                        features.append(temporal_features[k])
                        label.append(temporal_features[k.split("_")[0]+'_label'])                

        features = np.array(features)
        test_features = np.array(test_features)
        label = np.array(label).astype(int) 
        test_label = np.array(test_label).astype(int)
        return features, label, test_features, test_label

    def train(X, y):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare tensors
        X = np.array(X)
        y = np.array(y)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y - 1, dtype=torch.long).to(device)

        # DataLoader with shuffle
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=False)

        # Model, optimizer, loss
        model = FeatureMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        best_model = None
        best_train_acc = 0.0

        for epoch in range(1, 201):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            train_acc = correct / total * 100

            # Save best model by training accuracy
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_model = model.state_dict()
                # Optional logging
                # print(f"✅ Epoch {epoch:03d} | Train Acc Improved to {train_acc:.2f}% → Model Saved")

            # Optional: logging each epoch
            # print(f"Epoch {epoch:03d} | Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}%")

        return best_model

    
    def test(X, y, model_dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Adjust labels: [1, 2, 3] → [0, 1, 2]
        y_tensor = torch.tensor(y - 1, dtype=torch.long).to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        # Load model
        model = FeatureMLP().to(device)  # make sure this matches your training model
        model.load_state_dict(model_dict)
        model.eval()

        with torch.no_grad():
            logits_all= model(X_tensor)
            _, preds = torch.max(logits_all, 1)

        # Convert to numpy
        y_true = y_tensor.cpu().numpy()
        y_pred = preds.cpu().numpy()

        return y_true, y_pred
    
    def leave_one_out_train(path, operation):
        y_true = np.array([], dtype=int)
        y_pred = np.array([], dtype=int)

        for test_name in tqdm(video_dict.keys(), desc="LOO Fold"):  
            if operation == 1:
                X_train, y_train, X_test, y_test = leave_one_out.feature_loader(path, test_name)
            elif operation==2:
                X_train, y_train, X_test, y_test = leave_one_out.feature_loader2(path, test_name)
            elif operation==6:
                X_train, y_train, X_test, y_test = leave_one_out.feature_loader2(path, test_name)

            # y_test = smooth_labels(y_test,2)
            # y_train = smooth_labels(y_train,2)

            model = leave_one_out.train(X_train,y_train)
            true, pred = leave_one_out.test(X_test, y_test, model)

            
            print(f"-------------------------------------{test_name}--------------------------------------")

            cm = confusion_matrix(true, pred)
            print("Confusion Matrix:\n", cm)

            # Per-class accuracy
            for i in range(3):
                correct = np.sum((pred == i) & (true == i))
                total = np.sum(true == i)
                acc = correct / total * 100 if total > 0 else 0
                print(f"Class {i+1} Accuracy: {acc:.2f}%")

            # Classification report
            print("\nClassification Report:")
            print(classification_report(true, pred,
                    labels=[0, 1, 2],
                    target_names=["Class 1", "Class 2", "Class 3"],
                    zero_division=0))  # 避免除0错误


            y_true = np.concatenate([y_true, true]) if len(y_true) else true
            y_pred = np.concatenate([y_pred, pred]) if len(y_pred) else pred

        # Confusion matrix
        print("-------------------------------------fianl--------------------------------------")
        
        y_pred = smooth_labels(y_pred, 2)
        
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)

        # Per-class accuracy
        for i in range(3):
            correct = np.sum((y_pred == i) & (y_true == i))
            total = np.sum(y_true == i)
            acc = correct / total * 100 if total > 0 else 0
            print(f"Class {i+1} Accuracy: {acc:.2f}%")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
            labels=[0, 1, 2],
            target_names=["Class 1", "Class 2", "Class 3"],
            zero_division=0))


        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 1", "Class 2", "Class 3"], yticklabels=["Class 1", "Class 2", "Class 3"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
def main():
    # Assume you have:
    # X: (15164, 969) numpy array
    # y: (15164,) numpy array with 1, 2, 3 labels
    # X , y = pose_feature_loader("./dataset/facial_data")
    print( torch.cuda.is_available())
    operation=2
    if operation ==1:
        root_dir = "./features/combined_feature_with_reduced_pose_mlp"
        leave_one_out.leave_one_out_train(root_dir, operation)
    elif operation ==2:
        root_dir = "./features/combined_feature_with_reduced_pose_mlp_after_selection"
        leave_one_out.leave_one_out_train(root_dir, operation)
    elif operation ==3:
        root_dir = "./features/combined_feature_with_reduced_pose_mlp"
        half_half.half_half_train(root_dir, operation)
    elif operation ==4:
        root_dir = "./features/combined_feature_with_reduced_pose_mlp_after_selection"
        half_half.half_half_train(root_dir, operation)

    elif operation ==5:
        root_dir = "./features/combined_feature_with_reduced_pose_mlp_after_selection_with_mlp_half_half"
        half_half.half_half_train(root_dir, operation)
    elif operation ==6:
        root_dir = "./features/combined_feature_with_reduced_pose_mlp_after_selection_with_mlp_half_half"
        leave_one_out.leave_one_out_train(root_dir, 6)

if __name__ == "__main__":
    main()