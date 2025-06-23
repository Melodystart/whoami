const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const previewImg = document.getElementById("preview");
const goRegisterBtn = document.getElementById("goRegisterBtn");
const customFileBtn = document.getElementById("customFileBtn");
const fileInput = document.getElementById("fileInput");
const resultDiv = document.getElementById("result");

let isProcessing = false;
let latestProcessedImage = null;
let selfieSegmentation;
let recognizeInterval = null;
let useCamera = true;
let isCameraActive = false;
let currentAbortController = null;

function initSelfieSegmentation() {
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

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.fillStyle = "#1C2331";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    isCameraActive = true;

    async function processFrame() {
      if (!isCameraActive) return;
      if (video.readyState === 4) {
        await selfieSegmentation.send({ image: video });
      }
      requestAnimationFrame(processFrame);
    }

    processFrame();
  } catch (error) {
    console.error("無法開啟鏡頭:", error);
    resultDiv.textContent = "請開啟鏡頭，或切換為圖片上傳模式";
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

  latestProcessedImage = canvas.toDataURL("image/jpeg");
}

async function detectAndRecognize() {
  if (!latestProcessedImage || isProcessing) {
    return;
  }

  // 取消前一次的 fetch 請求
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

    if (faces.length === 0) {
      resultDiv.textContent = "未辨識到人臉";
    } else {
      resultDiv.innerHTML = faces
        .map((face) =>
          face.name
            ? `${face.name} (${face.similarity.toFixed(2)})`
            : "未知人員"
        )
        .join("<br>");
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
      resultDiv.textContent = "辨識失敗，請重新整理後再試";
    }
  } finally {
    isProcessing = false;
    currentAbortController = null;
  }
}

document.getElementById("customFileBtn").addEventListener("click", () => {
  document.getElementById("fileInput").click();
});

const modeToggleInput = document.getElementById("modeToggleInput");

modeToggleInput.addEventListener("change", () => {
  useCamera = !modeToggleInput.checked;

  previewImg.src = "";
  previewImg.style.display = "none";
  latestProcessedImage = null;
  isProcessing = false;

  if (useCamera) {
    canvas.style.display = "block";
    video.style.display = "block";
    customFileBtn.style.display = "none";
    resultDiv.textContent = "辨識中，請看向鏡頭";
    initSelfieSegmentation();

    // 啟動辨識
    if (recognizeInterval) clearInterval(recognizeInterval);
    recognizeInterval = setInterval(() => detectAndRecognize(), 1000);
  } else {
    canvas.style.display = "none";
    video.style.display = "none";
    customFileBtn.style.display = "inline-block";
    resultDiv.textContent = "";

    // 停止辨識
    if (recognizeInterval) {
      clearInterval(recognizeInterval);
      recognizeInterval = null;
    }

    // 停止攝影機串流
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

  useCamera = false;
  isProcessing = false;
  resultDiv.textContent = "辨識中，請稍候";

  const reader = new FileReader();
  reader.onload = (event) => {
    const imageDataUrl = event.target.result;

    const tempImg = new Image();
    tempImg.onload = async () => {
      previewImg.src = imageDataUrl;
      previewImg.style.display = "block";
      latestProcessedImage = imageDataUrl;
      await detectAndRecognize();
    };
    tempImg.src = imageDataUrl;
  };
  reader.readAsDataURL(file);
});

window.addEventListener("DOMContentLoaded", () => {
  modeToggleInput.dispatchEvent(new Event("change"));
});
