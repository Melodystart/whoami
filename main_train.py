import torch
import torch.nn as nn
from torchvision import models
from retinaface import RetinaFace
from torchvision import transforms
from datetime import datetime
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import io
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request, Form
from dotenv import get_key
from fastapi.responses import JSONResponse
import time
import asyncio
import gc
import boto3
from fastapi import BackgroundTasks
from databases import Database
import json 

DATABASE_URL = f"postgresql://{get_key('.env', 'USER')}:{get_key('.env', 'DB_PASSWORD')}@{get_key('.env', 'HOST')}:5432/{get_key('.env', 'DB_NAME')}"
database = Database(DATABASE_URL, min_size=5, max_size=20)

aws_access_key = get_key(".env", "aws_access_key")
aws_secret_key = get_key(".env", "aws_secret_key")
s3_bucket_name = get_key(".env", "s3_bucket_name")

s3 = boto3.client('s3', aws_access_key_id=aws_access_key,
                  aws_secret_access_key=aws_secret_key)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocess = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

db_pool = None
face_model = None
last_access_time = None
model_lock = asyncio.Lock()
# 限制同時只有 1 個推論，防止 CPU 過度負載
inference_semaphore = asyncio.Semaphore(1)

@app.on_event("startup")
async def startup_event():
    await database.connect()
    # 背景週期性檢查，若模型閒置10分鐘則解除模型引用並釋放記憶體
    asyncio.create_task(release_model_periodically())
    print("資料庫已連線")

async def insert_embedding(name, emb, bbox):
    emb_str = "[" + ",".join(map(str, emb.tolist())) + "]"
    sql = """
        INSERT INTO face_embeddings (filename, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        VALUES (:filename, :embedding, :x1, :y1, :x2, :y2)
    """
    values = {
        "filename": name,
        "embedding": emb_str,
        "x1": int(bbox[0]),
        "y1": int(bbox[1]),
        "x2": int(bbox[2]),
        "y2": int(bbox[3]),
    }
    await database.execute(query=sql, values=values)

# 計算兩向量相似度->用Sigmoid轉換0~1間的值並加入k調整Sigmoid轉換的敏感度
def cosine_sim_sigmoid(a, b, k=5):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    prob = 1 / (1 + np.exp(-k * cos_sim))
    return prob

async def get_embedding_and_bbox(np_image):
    async with inference_semaphore:  # 限制同時只有一個推論進入
        loop = asyncio.get_event_loop()
        model = await get_model()
        return await loop.run_in_executor(None, process_face, np_image, model)

def process_face(np_image, model):    
    img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]

    # 縮放比例（640 / 原圖最長邊）
    target_size = 640
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 縮放圖片（保持比例）
    resized_img = cv2.resize(img, (new_w, new_h))

    # 建立黑底 640×640 並置中貼圖
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded_img[top:top+new_h, left:left+new_w] = resized_img

    # 用縮放+padding後的圖片做偵測
    try:
        faces = RetinaFace.detect_faces(padded_img)
    except Exception as e:
        print(f"RetinaFace 偵測錯誤: {e}")
        return None, None
    
    if not isinstance(faces, dict) or len(faces) == 0:
        return None, None

    # 取分數最高的人臉
    best_face = max(faces.items(), key=lambda x: x[1]['score'])[1]
    x1_pad, y1_pad, x2_pad, y2_pad = map(int, best_face['facial_area'])

    # 把 bbox 從 640×640（含 padding）座標轉回原圖座標
    x1 = int((x1_pad - left) / scale)
    y1 = int((y1_pad - top) / scale)
    x2 = int((x2_pad - left) / scale)
    y2 = int((y2_pad - top) / scale)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None

    # 裁切原圖的人臉
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    # 轉成模型輸入格式
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_img = Image.fromarray(face_rgb)
    face_tensor = preprocess(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor).squeeze().cpu().numpy()

    return embedding, [x1, y1, x2, y2]

async def get_model():
    global face_model, last_access_time
    # asyncio.Lock 排隊機制，讓多個非同步請求能安全且有序地使用同一段關鍵程式碼
    # 避免多個請求同時開始重複載入模型
    async with model_lock:
        if face_model is None:
            try:
                print("載入模型中")
                model = torch.jit.load("train_best.pt").to(device)
                model.eval()
                face_model = model
                print("模型已載入")
            except Exception as e:
                print(f"模型載入失敗: {e}")
                face_model = None
                raise
        # 記錄模型最後一次被使用的時間，以便計算模型閒置時間
        last_access_time = time.time()
        return face_model

# 定期檢查模型是否閒置10分鐘，若是則解除模型引用並釋放記憶體
async def release_model_periodically():
    global face_model, last_access_time
    while True:
        await asyncio.sleep(60)
        if face_model and last_access_time and time.time() - last_access_time > 600:
            print("模型閒置10分鐘")
            # 解除模型引用
            face_model = None
            last_access_time = None
            # 記憶體釋放
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("模型已釋放")

async def search_similar_faces(query_emb, top_k=5):
    query_emb_str = '[' + ','.join(map(str, query_emb)) + ']'
    sql = """
        SELECT filename, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding, embedding <=> CAST(:query_emb AS vector) AS distance FROM face_embeddings ORDER BY distance ASC LIMIT :top_k
    """
    rows = await database.fetch_all(query=sql, values={"query_emb": query_emb_str, "top_k": top_k})
    return rows

def upload_to_s3(file_obj: io.BytesIO, filename: str):
    file_obj.seek(0)
    s3.upload_fileobj(file_obj, s3_bucket_name, filename)

@app.post("/api/register")
async def register(background_tasks: BackgroundTasks,name: str = Form(...), file: UploadFile = File(...)):
    
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"message": "檔案格式錯誤，請上傳圖片檔"},
        )
    
    start_time = time.perf_counter()

    t1 = time.perf_counter()
    content = await file.read()
    file_obj = io.BytesIO(content)
    t2 = time.perf_counter()

    # 將圖片轉成PIL圖片(RGB格式)
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        print(f"圖片轉成PIL圖片失敗: {e}")
        return JSONResponse(
            status_code=400,
            content={"message": "無法開啟圖片，請確認是否為有效圖片檔案"},
        )
    # 將PIL圖片轉為numpy array (H, W, C) 模型需要的格式
    np_image = np.array(image)
    t3 = time.perf_counter()

    emb, bbox = await get_embedding_and_bbox(np_image)
    t4 = time.perf_counter()
    if emb is None:
        return JSONResponse(status_code=400, content={"message": "未偵測到人臉"})

    now_str = datetime.now().strftime("%y%m%d%H%M%S")
    ext = os.path.splitext(file.filename)[1].lower()
    filename = f"{now_str}_{name}{ext}"
    background_tasks.add_task(upload_to_s3, file_obj, filename)
    t5 = time.perf_counter()

    await insert_embedding(name, emb, bbox)
    t6 = time.perf_counter()
    end_time = time.perf_counter()
    elapsed = round(end_time - start_time, 2)
    print(f"{name} 的照片已成功註冊")
    print(f"耗時: {elapsed} 秒")
    print(f"read_file: {t2 - t1:.2f} 秒")
    print(f"convert_image: {t3 - t2:.2f} 秒")
    print(f"face_detection: {t4 - t3:.2f} 秒")
    print(f"save_file: {t5 - t4:.2f} 秒")
    print(f"db_insert: {t6 - t5:.2f} 秒")
    return {"message": f"{name} 的照片已成功註冊", "filename": filename, "bbox":{"x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3]),}}

@app.post("/api/recognize")
async def recognize(background_tasks: BackgroundTasks, file: UploadFile = File(...),
    useCamera: str = Form(...)):
    content = await file.read()
    file_obj = io.BytesIO(content)
    now_str = datetime.now().strftime("%y%m%d%H%M%S")
    ext = os.path.splitext(file.filename)[1].lower()
    try:
        img_pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        print(f"圖片轉成PIL圖片失敗: {e}")
        return JSONResponse(
            status_code=400,
            content={"message": "無法開啟圖片，請確認是否為有效圖片檔案"},
        )
    np_image = np.array(img_pil)

    query_emb, query_bbox = await get_embedding_and_bbox(np_image)
    if query_emb is None:
        # filename = f"noface_{now_str}{ext}"
        return {"faces": [],"useCamera": useCamera}

    rows = await search_similar_faces(query_emb, top_k=5)

    scored_results = []
    for row in rows:
        fname, x1, y1, x2, y2, embedding, distance  = row["filename"], row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"], row["embedding"], row["distance"]
        embedding_list = json.loads(embedding)
        embedding_vec = np.array(embedding_list, dtype=np.float32)
        similarity = cosine_sim_sigmoid(query_emb, embedding_vec)
        scored_results.append((similarity, fname, x1, y1, x2, y2))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    best_similarity, best_match, x1, y1, x2, y2 = scored_results[0]

    print(f"找到相似人臉: {best_match}, 相似度: {best_similarity}")
    if best_similarity > 0.8:
        return {
            "faces": [{
                "x1": int(query_bbox[0]),
                "y1": int(query_bbox[1]),
                "x2": int(query_bbox[2]),
                "y2": int(query_bbox[3]),
                "name": best_match,
                "similarity": float(best_similarity)
            }],"useCamera": useCamera
        }
    else:
        filename = f"{best_similarity:.2f}".replace(".", "_") + f"_{best_match}_{now_str}{ext}"
        # background_tasks.add_task(upload_to_s3, file_obj, filename)
        print(f"相似度<0.8: {best_match}, 相似度: {best_similarity}")
        return {
            "faces": [{
                "x1": int(query_bbox[0]),
                "y1": int(query_bbox[1]),
                "x2": int(query_bbox[2]),
                "y2": int(query_bbox[3]),
                "name": None,
                "similarity": float(best_similarity)
            }],"useCamera": useCamera
        }

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="register.html"
    )

@app.get("/recognize", response_class=HTMLResponse)
async def recognize_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="recognize.html"
    )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.on_event("shutdown")
async def shutdown_event():
    await database.disconnect()
    print("資料庫連線池已關閉")