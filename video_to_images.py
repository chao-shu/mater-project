import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis
import csv
from insightface.utils.face_align import norm_crop
import numpy as np

def read_dense_frame_labels(fname):
    # Read label of specific student
    label_list = []
    with open(f"./data/label/denseframelabel.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == fname:
                label_list.append(row)
    
    return label_list



# Define a canonical template for alignment (e.g. 5-point eye/nose/mouth)
canonical_landmark_5pts = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)  # size: (5, 2), used by insightface in 112x112

def extract_faces_from_video_with_alignment(video_path, video_name, output_dir, face_app):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0
    ref_matrix = None  # Affine matrix computed on frame 3
    dense_frame_labels = read_dense_frame_labels(video_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= len(dense_frame_labels):
            break

        faces = face_app.get(frame)
        if len(faces) == 0:
            print(f"[Frame {frame_idx}] No face detected.")
            frame_idx += 1
            continue

        face = faces[0]
        if face.kps is None:
            print(f"[Frame {frame_idx}] No 5-point landmarks.")
            frame_idx += 1
            continue

        src_landmarks = face.kps.astype(np.float32)  # (5, 2)

        if frame_idx == 3:
            # compute affine transform from landmarks to canonical template
            ref_matrix, _ = cv2.estimateAffinePartial2D(src_landmarks, canonical_landmark_5pts)
            if ref_matrix is None:
                print("Failed to compute reference affine transform.")
                break
            print(f"[Frame 3] Alignment matrix computed.")

        if ref_matrix is None:
            x1, y1, x2, y2 = [int(x) for x in face.bbox]
            aligned_face = frame[y1:y2, x1:x2]
            aligned_face = cv2.resize(aligned_face,(224,224))
            out_path = os.path.join(output_dir, f"face_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, aligned_face)
            saved_count += 1
            frame_idx += 1
            continue

        # align face using the fixed matrix
        aligned_face = cv2.warpAffine(frame, ref_matrix, (112, 112))
        aligned_face = cv2.resize(aligned_face,(224,224))
        out_path = os.path.join(output_dir, f"face_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, aligned_face)
        saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_count


def extract_faces_from_video(video_path, video_name, output_dir, face_or_frame, max_faces=None):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0

    dense_frame_labels = read_dense_frame_labels(video_name)

    if face_or_frame == "face":
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= len(dense_frame_labels):
                break
            faces = face_app.get(frame)
            if len(faces) > 0:
                face = faces[0]
                x1, y1, x2, y2 = [int(x) for x in face.bbox]
                aligned_face = frame[y1:y2, x1:x2]

                if aligned_face.size == 0:
                    frame_idx += 1
                    continue
                
                # resize成224x224
                aligned_face = cv2.resize(aligned_face, (224, 224))

                out_path = os.path.join(output_dir, f"face_{frame_idx:05d}.jpg")
                cv2.imwrite(out_path, aligned_face)
                saved_count += 1

            frame_idx += 1
            if max_faces is not None and saved_count >= max_faces:
                break
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= len(dense_frame_labels):
                break
            out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1

            frame_idx += 1
            if max_faces is not None and saved_count >= max_faces:
                break

    cap.release()
    return saved_count
video = [
    "1658828085789384", "165745616715997", "16572216818408213", "16580176851586106",
    "1657019500197831", "1657025477939515", "16569562540764549", "16570365844633188",
    "16571735778363602", "16572034690729465", "16643616205681827", "16638490487013648",
    "16637850973931577", "16637915350525117", "16639504750710998", "16638397654914925",
    "16400282145", "163904405748", "163904657409", "163965258378",
    "163974758282", "164002054426", "164006412052", "164007913498",
    "164008493608"
]
if __name__ == "__main__":
    raw_video_dir = "./data/converted"
    output_root = "./data/face_with_alignment"
    face_or_frame = "face"  # or "face"
    os.makedirs(output_root, exist_ok=True)

    # 初始化 RetinaFace
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    face_app.prepare(ctx_id=0)
    video_list = [f for f in os.listdir(raw_video_dir) if f.endswith(('.mp4'))]

    for video_file in tqdm(video_list, desc="Processing videos"):
        video_path = os.path.join(raw_video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_root, video_name)
        if video_name in video:
            # count = extract_faces_from_video(video_path, video_name, output_dir, face_or_frame)
            count = extract_faces_from_video_with_alignment(video_path, video_name, output_dir, face_app)
            print(f"{video_name}: saved {count} face frames.")
