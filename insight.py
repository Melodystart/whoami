import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
import torch
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

arcface = FaceAnalysis(name='antelopev2')
ctx_id = 0 if torch.cuda.is_available() else -1

# 計算cosine相似度後用Sigmoid將輸出轉為0~1的值，k調整Sigmoid敏感度，讓「相似」和「非常相似」的差異更明顯
def cosine_sim_sigmoid(a, b, k=5):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    prob = 1 / (1 + np.exp(-k * cos_sim))
    return prob

def get_embedding_and_bbox(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"讀取圖片失敗: {image_path}")
        return None, None, None

    h, w = img.shape[:2]

    # 動態決定det_size，最大邊長限制640
    max_side = max(w, h)
    max_det_size = 640
    scale = 1.0
    if max_side > max_det_size:
        scale = max_det_size / max_side
    det_w = int(w * scale)
    det_h = int(h * scale)

    # InsightFace的det_size須是32的倍數
    def make_divisible(x, divisor=32):
        return x if x % divisor == 0 else (x // divisor) * divisor

    det_w = make_divisible(det_w)
    det_h = make_divisible(det_h)

    # 若模型對某張臉的信心(det_thresh)低於 0.5，就會忽略該人臉
    arcface.prepare(ctx_id=ctx_id, det_size=(det_w, det_h), det_thresh=0.5)

    # 若縮放過圖片，偵測時先resize圖片，並把bbox放大回原圖尺寸
    if scale != 1.0:
        img_resized = cv2.resize(img, (det_w, det_h))
        faces = arcface.get(img_resized)
        if len(faces) == 0:
            print(f"InsightFace 沒偵測到臉: {image_path}")
            return None, None, None

        # 依 det_score 信心分數由高到低排序後，取信心最高的人臉
        faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
        face = faces[0]
        embedding = face.embedding
        bbox = face.bbox.astype(float)

        # bbox放大回原始尺寸
        bbox[0] = bbox[0] / scale
        bbox[1] = bbox[1] / scale
        bbox[2] = bbox[2] / scale
        bbox[3] = bbox[3] / scale
        bbox = bbox.astype(int)
    else:
        faces = arcface.get(img)
        if len(faces) == 0:
            print(f"InsightFace 沒偵測到臉: {image_path}")
            return None, None, None

        faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
        face = faces[0]
        embedding = face.embedding
        bbox = face.bbox.astype(int)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    return embedding, bbox, img_pil

# database資料夾：模擬每個員工各一張入職照片
database_path = './database'
db_embeddings = []
db_bboxes = []
db_images = []
db_filenames = []

for filename in sorted(os.listdir(database_path)):
    image_path = os.path.join(database_path, filename)
    if not os.path.isfile(image_path):
        continue
    emb, box, img = get_embedding_and_bbox(image_path)
    if emb is None:
        continue
    db_embeddings.append(emb)
    db_bboxes.append(box.tolist())
    db_images.append(img)
    db_filenames.append(filename)
    print(f"V 建立 embedding: {filename}")

# query資料夾：測試辨識結果的圖片
query_folder = './query'
query_filenames = sorted([f for f in os.listdir(query_folder) if os.path.isfile(os.path.join(query_folder, f))])

for query_filename in query_filenames:
    query_image_path = os.path.join(query_folder, query_filename)
    query_emb, query_box, query_img = get_embedding_and_bbox(query_image_path)
    if query_emb is None:
        print(f"Query 圖片無法偵測臉部: {query_filename}")
        continue

    prob_list = []
    for emb in db_embeddings:
        prob = cosine_sim_sigmoid(query_emb, emb, k=5)
        prob_list.append(prob)

    combined = list(zip(prob_list, db_embeddings, db_bboxes, db_images, db_filenames))
    combined.sort(key=lambda x: x[0], reverse=True)

    # 顯示前5名
    top_k = 5
    top_combined = combined[:top_k]
    fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 5))
    if not isinstance(axes, (np.ndarray, list)):
        axes = [axes]
    plt.suptitle(f"Query: {query_filename}")

    # 測試辨識的圖片放第一張圖的位置
    ax = axes[0]

    if isinstance(query_img, np.ndarray):
        img_copy = Image.fromarray(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    else:
        img_copy = query_img.copy()
    box_coords = [int(x) for x in query_box.tolist()]
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(box_coords, outline='red', width=3)
    
    ax.imshow(img_copy)
    ax.set_title("Test Image")
    ax.axis('off')

    # 依相似度排序顯示資料庫圖片
    for i, (prob, emb, box, img, fname) in enumerate(top_combined):
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle(box, outline='green', width=3)
        ax = axes[i + 1]
        ax.imshow(img_copy)
        ax.set_title(f"{fname}\nSimilarity: {prob:.4f}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    best_prob = combined[0][0]
    best_match = combined[0][4]    

    print(f"最佳比對結果: {best_match} (相似度: {best_prob:.4f})")
    if best_prob > 0.6:
        print(f"找到相似人臉: {best_match}")
    else:
        print("查無相似人臉")