import torch
import torch.nn as nn

class FeatureMLP(nn.Module):
    def __init__(self):
        super(FeatureMLP, self).__init__()
        self.fc1 = nn.Linear(969,512)
        self.relu1 = nn.ReLU()
        self.Dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.relu2(self.fc2(self.Dropout1(self.relu1(self.fc1(x)))))
        feat_128 = self.relu3(self.fc3(x))
        logits = self.fc4(feat_128)
        return logits, feat_128
    


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
def balance_dataset(X, y):
    X = np.array(X)
    y = np.array(y)

    # Count class distribution
    counter = Counter(y)
    print(f"Original class distribution: {counter}")

    # Find minimum class count
    min_count = min(counter.values())

    # Collect balanced indices
    indices = []
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        selected = np.random.choice(label_indices, min_count, replace=False)
        indices.extend(selected)

    # Shuffle and return balanced dataset
    np.random.shuffle(indices)
    return X[indices], y[indices]

def train(X, y, model_path="./checkpoint/pose_feature_mlp.pth"):
    # Prepare tensors
    X = np.array(X)
    y = np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y - 1, dtype=torch.long)  # shift labels to [0,1,2]

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    # Model
    model = FeatureMLP()
    optimizer = torch.optim.Adam(model.parameters())

    # Compute class weights


    # Weighted CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total * 100

        # Evaluate on validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits, _ = model(batch_x)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        val_acc = val_correct / val_total * 100

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"✅ Epoch {epoch:02d} | Val Acc Improved to {val_acc:.2f}% → Model Saved")

        print(f"Epoch {epoch:02d} | Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

def test(X, y, model_path="./checkpoint/pose_feature_mlp.pth"):
    # Adjust labels: [1, 2, 3] → [0, 1, 2]
    y_tensor = torch.tensor(y - 1, dtype=torch.long)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Load model
    model = FeatureMLP()  # make sure this matches your training model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        logits_all, _ = model(X_tensor)
        _, preds = torch.max(logits_all, 1)

    # Convert to numpy
    y_true = y_tensor.numpy()
    y_pred = preds.numpy()

    # Confusion matrix
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
    print(classification_report(y_true, y_pred, target_names=["Class 1", "Class 2", "Class 3"]))

def save_feature(root_dir, model_path="./checkpoint/pose_feature_mlp.pth", save_dir="./features/pose_mlp_feature"):
    # Load trained model
    model = FeatureMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    video_names = sorted(os.listdir(root_dir))
    for video_name in video_names:
        video_feature_path = os.path.join(root_dir, video_name)

        with open(video_feature_path, 'rb') as f:
            pose_features = pickle.load(f)

        frame_ids = sorted(pose_features.keys())
        feature_dict = {}

        for k in frame_ids:
            if k.split("_")[1] == "face":
                input_tensor = torch.tensor(pose_features[k], dtype=torch.float32).unsqueeze(0)  # shape (1, 969)
                with torch.no_grad():
                    _, mid_feature = model(input_tensor)
                    mid_feature_np = mid_feature.squeeze(0).numpy()

                # ✅ Check if the feature has any NaN
                if np.isnan(mid_feature_np).any():
                    print(f"⚠️ NaN detected in feature for frame: {k} of video: {video_name}")
                    continue  # Skip this frame

                feature_dict[k] = mid_feature_np
                feature_dict[k.split("_")[0]+"_label"] = pose_features[k.split("_")[0]+"_label"]
        save_path = os.path.join(save_dir, f"{video_name}")
        with open(save_path, "wb") as f:
            pickle.dump(feature_dict, f)

        print(f"✅ Saved features to: {save_path}")

def main():
    # Assume you have:
    # X: (15164, 969) numpy array
    # y: (15164,) numpy array with 1, 2, 3 labels
    # X , y = pose_feature_loader("./dataset/facial_data")
    root_dir = "./features/normalized_pose_feature"
    X,y = reduced_pose_feature_loader(root_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    operation =2
    if operation == 1:
        train(X_train, y_train)
    elif operation ==2:
        test(X_test,y_test)
    elif operation ==3:
        save_feature(root_dir)



if __name__ == "__main__":
    main()