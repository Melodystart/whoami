const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let isProcessing = false;
let latestProcessedImage = null;

const selfieSegmentation = new SelfieSegmentation({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
});

selfieSegmentation.setOptions({
  modelSelection: 1,
});

selfieSegmentation.onResults(onResults);

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
  });
  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  ctx.fillStyle = "#1C2331";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  async function processFrame() {
    if (video.readyState === 4) {
      await selfieSegmentation.send({ image: video });
    }
    requestAnimationFrame(processFrame);
  }

  processFrame();
}

function onResults(results) {
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

const resultDiv = document.getElementById("result");

async function detectAndRecognize() {
  if (!latestProcessedImage || isProcessing) return;

  isProcessing = true;
  try {
    const t1 = performance.now();
    const blob = await (await fetch(latestProcessedImage)).blob();
    const t2 = performance.now();
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");
    const t3 = performance.now();

    const res = await fetch("/api/recognize", {
      method: "POST",
      body: formData,
    });
    const t4 = performance.now();
    const result = await res.json();
    const faces = result.faces || [];
    const t5 = performance.now();
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
    console.error("辨識錯誤:", err);
  } finally {
    isProcessing = false;
  }
}

setInterval(detectAndRecognize, 1000);

startCamera();
