from pydantic import BaseModel
from datetime import datetime
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import io
import ast
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request, Form
from insightface.app import FaceAnalysis
import torch
from dotenv import get_key
from fastapi.responses import JSONResponse
import time
from psycopg2 import pool
import asyncio
import gc
import boto3

aws_access_key = get_key(".env", "aws_access_key")
aws_secret_key = get_key(".env", "aws_secret_key")
s3_bucket_name = get_key(".env", "s3_bucket_name")

s3 = boto3.client('s3', aws_access_key_id=aws_access_key,
                  aws_secret_access_key=aws_secret_key)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx_id = 0 if torch.cuda.is_available() else -1

db_pool = None
arcface = None
last_access_time = None
model_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global db_pool
    asyncio.create_task(release_model_periodically())

    try:
        db_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            dbname=get_key(".env", "DB_NAME"),
            user=get_key(".env", "USER"),
            password=get_key(".env", "DB_PASSWORD"),
            host=get_key(".env", "HOST"),
            port="5432"
        )
        if db_pool:
            print("資料庫連線池已建立")
    except Exception as e:
        print("建立資料庫連線池失敗:", e)
        raise

def insert_embedding(name, emb, bbox):
    emb_list = emb.tolist()
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO face_embeddings (filename, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (name, emb_list, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            conn.commit()
    except Exception as e:
        print(f"插入資料庫錯誤: {e}")
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)

def cosine_sim_sigmoid(a, b, k=5):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    prob = 1 / (1 + np.exp(-k * cos_sim))
    return prob

async def get_embedding_and_bbox(np_image):
    # h, w = np_image.shape[:2]
    # max_side = max(h, w)
    # max_det_size = 640
    # scale = 1.0
    # img = np_image.copy()
    # if max_side > max_det_size:
    #     scale = max_det_size / max_side
    #     img = cv2.resize(img, (int(w * scale), int(h * scale)))

    arcface = await get_arcface_model()
    # faces = arcface.get(img)
    faces = arcface.get(np_image)
    if len(faces) == 0:
        return None, None

    faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
    face = faces[0]
    embedding = face.embedding
    bbox = face.bbox.astype(float)

    # if scale != 1.0:
    #     bbox /= scale

    bbox = bbox.astype(int)
    return embedding, bbox

async def get_arcface_model():
    global arcface, last_access_time
    async with model_lock:
        if arcface is None:
            try:
                print("載入模型中")
                arcface = FaceAnalysis(name='antelopev2')
                arcface.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.5)
                print("模型已載入")
            except Exception as e:
                print(f"模型載入失敗: {e}")
                arcface = None
                raise
        last_access_time = time.time()
        return arcface

async def release_model_periodically():
    global arcface, last_access_time
    while True:
        await asyncio.sleep(60)
        if arcface and last_access_time and time.time() - last_access_time > 600:
            print("模型閒置10分鐘")
            arcface = None
            last_access_time = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("模型已釋放")

def search_similar_faces(query_emb, top_k=5):
    query_emb_str = '[' + ','.join(map(str, query_emb)) + ']'
    sql = """
    SELECT filename, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding, embedding <-> CAST(%s AS vector) AS distance
    FROM face_embeddings
    ORDER BY distance
    LIMIT %s;
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (query_emb_str, top_k))
            results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"查詢資料庫錯誤: {e}")
        raise
    finally:
        db_pool.putconn(conn)
        
@app.post("/api/register")
async def register(name: str = Form(...), file: UploadFile = File(...)):
    start_time = time.perf_counter()

    t1 = time.perf_counter()
    content = await file.read()
    file_obj = io.BytesIO(content)
    t2 = time.perf_counter()

    image = Image.open(io.BytesIO(content)).convert("RGB")
    np_image = np.array(image)
    t3 = time.perf_counter()

    emb, bbox = await get_embedding_and_bbox(np_image)
    t4 = time.perf_counter()
    if emb is None:
        return JSONResponse(status_code=400, content={"message": "未偵測到人臉"})

    now_str = datetime.now().strftime("%y%m%d%H%M%S")
    ext = os.path.splitext(file.filename)[1].lower()
    filename = f"{now_str}_{name}{ext}"
    s3.upload_fileobj(file_obj, s3_bucket_name, filename)
    t5 = time.perf_counter()

    insert_embedding(name, emb, bbox)
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
    return {"message": f"{name} 的照片已成功註冊", "filename": filename}

@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...),
    useCamera: str = Form(...)):
    content = await file.read()
    file_obj = io.BytesIO(content)
    now_str = datetime.now().strftime("%y%m%d%H%M%S")
    ext = os.path.splitext(file.filename)[1].lower()
    img_pil = Image.open(io.BytesIO(content)).convert("RGB")
    np_image = np.array(img_pil)

    query_emb, query_box = await get_embedding_and_bbox(np_image)
    if query_emb is None:
        # filename = f"noface_{now_str}{ext}"
        # s3.upload_fileobj(file_obj, s3_bucket_name, filename)
        return {"faces": [],"useCamera": useCamera}

    results = search_similar_faces(query_emb, top_k=5)

    scored_results = []
    for fname, x1, y1, x2, y2, embedding, distance in results:
        embedding_list = ast.literal_eval(embedding)
        embedding_vec = np.array(embedding_list, dtype=np.float32)
        similarity = cosine_sim_sigmoid(query_emb, embedding_vec)
        scored_results.append((similarity, fname, x1, y1, x2, y2))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    best_similarity, best_match, x1, y1, x2, y2 = scored_results[0]

    print(f"找到相似人臉: {best_match}, 相似度: {best_similarity}")
    if best_similarity > 0.8:
        return {
            "faces": [{
                "x1": int(query_box[0]),
                "y1": int(query_box[1]),
                "x2": int(query_box[2]),
                "y2": int(query_box[3]),
                "name": best_match,
                "similarity": float(best_similarity)
            }],"useCamera": useCamera
        }
    else:
        filename = f"{best_similarity:.2f}".replace(".", "_") + f"_{best_match}_{now_str}{ext}"
        s3.upload_fileobj(file_obj, s3_bucket_name, filename)
        return {
            "faces": [{
                "x1": int(query_box[0]),
                "y1": int(query_box[1]),
                "x2": int(query_box[2]),
                "y2": int(query_box[3]),
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
def shutdown_event():
    if db_pool:
        db_pool.closeall()
        print("資料庫連線池已關閉")