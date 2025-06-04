import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 人臉偵測，keep_all=False只回傳最大或最清楚的一張臉
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
# 人臉轉成特徵向量
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 計算cosine相似度後用Sigmoid將輸出轉為0~1的值，k調整Sigmoid敏感度，讓「相似」和「非常相似」的差異更明顯
def cosine_sim_sigmoid(a, b, k=5):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    prob = 1 / (1 + np.exp(-k * cos_sim))
    return prob

def get_embedding_and_bbox(image_path):
    img = Image.open(image_path).convert('RGB')
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        print(f"MTCNN 沒偵測到臉: {image_path}")
        return None, None, None

    box = boxes[0]
    # 方框的位置資訊（數字座標）
    boxes_tensor = torch.tensor([box], dtype=torch.float32)
    # 方框裁剪出來的臉部影像（像素資料）
    face_tensor = mtcnn.extract(img, boxes_tensor, save_path=None).to(device)
    face_tensor = face_tensor.unsqueeze(0)  # 加上 batch 維度，因 keep_all=False 只回傳一張臉 → shape 為 [1, channels, height, width]
    with torch.no_grad():
        embedding = facenet(face_tensor).cpu().numpy()[0]
    return embedding, box, img

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
    axes = np.array(axes).flatten() 
    # fig, axes = plt.subplots(1, len(db_embeddings) + 1, figsize=(4 * (len(db_embeddings) + 1), 5))
    if not isinstance(axes, (np.ndarray, list)):
        axes = [axes]
    plt.suptitle(f"Query: {query_filename}")

    # 測試辨識的圖片放第一張圖的位置
    ax = axes[0]
    
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
