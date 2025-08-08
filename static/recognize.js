const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const canvasPreview = document.getElementById("preview");
const ctxPreview = canvasPreview.getContext("2d");
const goRegisterBtn = document.getElementById("goRegisterBtn");
const customFileBtn = document.getElementById("customFileBtn");
const fileInput = document.getElementById("fileInput");
const resultDiv = document.getElementById("result");
const modeToggleInput = document.getElementById("modeToggleInput");

let isProcessing = false;
let latestProcessedImage = null;
let selfieSegmentation;
let recognizeInterval = null;
let useCamera = true;
let isCameraActive = false;
let currentAbortController = null;
let originalImageWidth = null;
let originalImageHeight = null;

// 閒置控制，只有在使用攝影模式時才有用
let idleTimeout = null;
let isIdle = false;
let wasRecognizing = false;
const IDLE_TIME = 60000;

function resetIdleTimer() {
  if (!useCamera) return;
  clearTimeout(idleTimeout);
  if (isIdle) {
    resumeRecognition();
  }
  idleTimeout = setTimeout(() => {
    pauseRecognition("閒置");
  }, IDLE_TIME);
}

function pauseRecognition(reason = "未知原因") {
  if (!useCamera) return;

  if (recognizeInterval) {
    clearInterval(recognizeInterval);
    recognizeInterval = null;
  }
  if (!isIdle) {
    wasRecognizing = useCamera && isCameraActive;
    isIdle = true;
    isCameraActive = false;
    canvas.style.display = "none";
    resultDiv.textContent = `辨識暫停(${reason})，請移動滑鼠恢復辨識`;
    console.log(`辨識暫停（${reason}）`);
  }
}

function resumeRecognition() {
  if (!useCamera) return;

  if (isIdle && wasRecognizing) {
    isCameraActive = true;
    canvas.style.display = "block";
    recognizeInterval = setInterval(() => detectAndRecognize(), 4000);
    resultDiv.textContent = "辨識中，請看向鏡頭";
    isIdle = false;
    console.log("辨識已恢復");
  }
}

async function initSelfieSegmentation() {
  if (selfieSegmentation) {
    await selfieSegmentation.close();
  }

  selfieSegmentation = new SelfieSegmentation({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
  });

  selfieSegmentation.setOptions({
    modelSelection: 1,
  });

  selfieSegmentation.onResults(onResults);
  startCamera();
}

let isSending = false;

async function processFrame() {
  if (
    video.readyState >= 2 &&
    video.videoWidth > 0 &&
    video.videoHeight > 0 &&
    isCameraActive &&
    selfieSegmentation
  ) {
    if (!isSending) {
      isSending = true;
      try {
        await selfieSegmentation.send({ image: video });
      } catch (e) {
        console.error("selfieSegmentation.send 錯誤:", e);
      } finally {
        isSending = false;
      }
    }
  }
  requestAnimationFrame(processFrame);
}

async function startCamera() {
  try {
    if (isCameraActive) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
    });
    video.srcObject = stream;
    await video.play();

    const width = 360;
    const height = 270;

    canvas.width = width;
    canvas.height = height;

    video.width = width;
    video.height = height;

    ctx.fillStyle = "#1C2331";
    ctx.fillRect(0, 0, width, height);

    isCameraActive = true;

    processFrame();
  } catch (error) {
    resultDiv.textContent = "請開啟攝影機，或切換為圖片上傳模式";
    isCameraActive = false;
  }
}

function onResults(results) {
  if (!useCamera) return;

  canvas.width = results.image.width;
  canvas.height = results.image.height;

  ctx.fillStyle = "#1C2331";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = canvas.width;
  tmpCanvas.height = canvas.height;
  const tmpCtx = tmpCanvas.getContext("2d");

  tmpCtx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
  tmpCtx.globalCompositeOperation = "destination-in";
  tmpCtx.drawImage(results.segmentationMask, 0, 0, canvas.width, canvas.height);
  ctx.drawImage(tmpCanvas, 0, 0);

  ctx.globalCompositeOperation = "destination-over";
  ctx.fillStyle = "#1C2331";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "source-over";

  latestProcessedImage = canvas.toDataURL("image/jpeg");
}

async function detectAndRecognize() {
  if (!latestProcessedImage || isProcessing) {
    return;
  }

  if (isIdle) {
    console.log("略過辨識流程(已暫停)");
    return;
  }

  // 取消前次fetch 請求，避免重複請求
  if (currentAbortController) {
    currentAbortController.abort();
  }
  const abortController = new AbortController();
  currentAbortController = abortController;

  isProcessing = true;

  try {
    const t1 = performance.now();
    const blob = await (await fetch(latestProcessedImage)).blob();
    const t2 = performance.now();
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");
    formData.append("useCamera", useCamera ? "true" : "false");
    const t3 = performance.now();

    const res = await fetch("/api/recognize", {
      method: "POST",
      body: formData,
      signal: abortController.signal,
    });
    const t4 = performance.now();
    const result = await res.json();
    const faces = result.faces || [];
    const beforeUseCamera = result.useCamera || "";
    const t5 = performance.now();

    if (useCamera.toString() !== beforeUseCamera.toString()) {
      console.warn("模式已切換，捨棄此辨識結果");
      return;
    }

    if (useCamera && isIdle) {
      console.log("辨識已暫停，捨棄此辨識結果");
      return;
    }

    if (faces.length === 0) {
      resultDiv.textContent = "未辨識到人臉";
      if (!useCamera) {
        ctxPreview.clearRect(0, 0, canvasPreview.width, canvasPreview.height);
        const img = new Image();
        img.onload = () => {
          const scale = Math.min(
            canvasPreview.width / originalImageWidth,
            canvasPreview.height / originalImageHeight
          );
          const imgWidth = originalImageWidth * scale;
          const imgHeight = originalImageHeight * scale;
          const offsetX = (canvasPreview.width - imgWidth) / 2;
          const offsetY = (canvasPreview.height - imgHeight) / 2;

          ctxPreview.drawImage(img, offsetX, offsetY, imgWidth, imgHeight);
        };
        img.src = latestProcessedImage;
      }
    } else {
      resultDiv.innerHTML = faces
        .map((face) =>
          face.name
            ? `${face.name} (${face.similarity.toFixed(2)})`
            : "未知人員"
        )
        .join("<br>");
      if (!useCamera) {
        const img = new Image();
        img.onload = () => {
          ctxPreview.clearRect(0, 0, canvasPreview.width, canvasPreview.height);
          const scale = Math.min(
            canvasPreview.width / originalImageWidth,
            canvasPreview.height / originalImageHeight
          );
          const imgWidth = originalImageWidth * scale;
          const imgHeight = originalImageHeight * scale;
          const offsetX = (canvasPreview.width - imgWidth) / 2;
          const offsetY = (canvasPreview.height - imgHeight) / 2;

          ctxPreview.drawImage(img, offsetX, offsetY, imgWidth, imgHeight);

          faces.forEach((face) => {
            const x = face.x1 * scale + offsetX;
            const y = face.y1 * scale + offsetY;
            const w = (face.x2 - face.x1) * scale;
            const h = (face.y2 - face.y1) * scale;

            ctxPreview.beginPath();
            ctxPreview.strokeStyle = face.name ? "lime" : "red";
            ctxPreview.lineWidth = 3;
            ctxPreview.rect(x, y, w, h);
            ctxPreview.stroke();

            ctxPreview.fillStyle = face.name ? "lime" : "red";
            ctxPreview.font = "16px Arial";
            ctxPreview.fillText(
              face.name
                ? `${face.name} (${face.similarity.toFixed(2)})`
                : "未知人員",
              x,
              y - 10
            );
          });
        };
        img.src = latestProcessedImage;
      }
    }
    console.log(`blob 轉換耗時: ${(t2 - t1).toFixed(1)} ms`);
    console.log(`發送至後端耗時: ${(t4 - t3).toFixed(1)} ms`);
    console.log(`回傳解析耗時: ${(t5 - t4).toFixed(1)} ms`);
    console.log(`總耗時: ${(t5 - t1).toFixed(1)} ms`);
  } catch (err) {
    if (err.name === "AbortError") {
      console.warn("請求已被中止");
    } else {
      console.error("辨識錯誤:", err);
      resultDiv.textContent = "辨識失敗，請重新整理或圖片格式無法讀取";
    }
  } finally {
    isProcessing = false;
    currentAbortController = null;
  }
}

customFileBtn.addEventListener("click", () => {
  document.getElementById("fileInput").click();
});

modeToggleInput.addEventListener("change", async () => {
  if (currentAbortController) {
    currentAbortController.abort();
  }
  useCamera = !modeToggleInput.checked;

  canvasPreview.style.display = "none";
  latestProcessedImage = null;
  isProcessing = false;

  if (useCamera) {
    canvas.style.display = "block";
    customFileBtn.style.display = "none";
    resultDiv.textContent = "辨識中，請看向鏡頭";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#1C2331";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    await initSelfieSegmentation();

    if (recognizeInterval) clearInterval(recognizeInterval);
    recognizeInterval = setInterval(() => detectAndRecognize(), 4000);
  } else {
    canvas.style.display = "none";
    customFileBtn.style.display = "inline-block";
    resultDiv.textContent = "";

    if (recognizeInterval) {
      clearInterval(recognizeInterval);
      recognizeInterval = null;
    }

    const stream = video.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    }
    isCameraActive = false;
  }
});

goRegisterBtn.addEventListener("click", () => {
  window.location.href = "/register";
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    resultDiv.textContent = "請選擇圖片檔案";
    return;
  }

  customFileBtn.disabled = true;
  customFileBtn.style.cursor = "not-allowed";
  useCamera = false;
  isProcessing = false;
  resultDiv.textContent = "";
  customFileBtn.textContent = "辨識中，請稍候";

  const reader = new FileReader();
  reader.onload = (event) => {
    const imageDataUrl = event.target.result;

    const tempImg = new Image();
    tempImg.onerror = () => {
      resultDiv.textContent = "圖片載入失敗，請重新選擇不同檔案";
      customFileBtn.disabled = false;
      customFileBtn.textContent = "選擇圖片";
    };
    tempImg.onload = async () => {
      canvasPreview.style.display = "block";

      canvasPreview.width = 360;
      canvasPreview.height = 270;
      originalImageWidth = tempImg.width;
      originalImageHeight = tempImg.height;

      ctxPreview.clearRect(0, 0, canvasPreview.width, canvasPreview.height);
      const scale = Math.min(
        canvasPreview.width / tempImg.width,
        canvasPreview.height / tempImg.height
      );
      const imgWidth = tempImg.width * scale;
      const imgHeight = tempImg.height * scale;
      const offsetX = (canvasPreview.width - imgWidth) / 2;
      const offsetY = (canvasPreview.height - imgHeight) / 2;

      ctxPreview.drawImage(tempImg, offsetX, offsetY, imgWidth, imgHeight);

      latestProcessedImage = imageDataUrl;
      await detectAndRecognize();
      customFileBtn.disabled = false;
      customFileBtn.style.cursor = "pointer";
      customFileBtn.textContent = "選擇圖片";
    };
    tempImg.src = imageDataUrl;
  };
  reader.readAsDataURL(file);
  e.target.value = "";
});

window.addEventListener("DOMContentLoaded", () => {
  modeToggleInput.dispatchEvent(new Event("change"));
  resetIdleTimer();
});

["mousemove", "keydown", "touchstart"].forEach((evt) =>
  window.addEventListener(evt, resetIdleTimer)
);

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    pauseRecognition("分頁切換");
  } else {
    resetIdleTimer();
  }
});
