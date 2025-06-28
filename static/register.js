const video = document.createElement("video");
video.style.display = "none";
document.body.appendChild(video);

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const canvasPreview = document.getElementById("preview");
const ctxPreview = canvasPreview.getContext("2d");
const registerBtn = document.getElementById("registerBtn");
const fileInput = document.getElementById("fileInput");
const customFileBtn = document.getElementById("customFileBtn");
const takePhotoBtn = document.getElementById("takePhotoBtn");
const modeToggleBtn = document.getElementById("modeToggleBtn");
const nameInput = document.getElementById("nameInput");
const goRecognizeBtn = document.getElementById("goRecognizeBtn");

let useCamera = true;
let selfieSegmentation = null;
let camera = null;
let hasPreviewImage = false;

navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
    initSelfieSegmentation();
  })
  .catch((err) => {
    console.error("無法取得攝影機:", err);
    alert("請開啟攝影機，或切換為圖片上傳模式");
  });

function initSelfieSegmentation() {
  if (selfieSegmentation || camera) return;
  selfieSegmentation = new SelfieSegmentation({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
  });

  selfieSegmentation.setOptions({ modelSelection: 1 });
  selfieSegmentation.onResults(onResults);

  selfieSegmentation.initialize().then(() => {
    camera = new Camera(video, {
      onFrame: async () => {
        if (selfieSegmentation) {
          await selfieSegmentation.send({ image: video });
        }
      },
      width: 360,
      height: 270,
    });

    camera.start();
  });
}

function stopSegmentation() {
  if (camera) {
    camera.stop();
    camera = null;
  }

  if (selfieSegmentation) {
    if (typeof selfieSegmentation.close === "function") {
      selfieSegmentation.close();
    }
    selfieSegmentation = null;
  }
}

function onResults(results) {
  canvas.width = results.image.width;
  canvas.height = results.image.height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
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
}

async function register() {
  registerBtn.disabled = true;
  registerBtn.style.cursor = "not-allowed";
  const name = document.getElementById("nameInput").value.trim();
  if (!name) {
    alert("請輸入姓名");
    registerBtn.disabled = false;
    registerBtn.style.cursor = "pointer";
    return;
  }

  if (!hasPreviewImage) {
    alert("請先拍照或上傳圖片");
    registerBtn.disabled = false;
    registerBtn.style.cursor = "pointer";
    return;
  }

  const dataURL = canvasPreview.toDataURL("image/png");
  const blob = await (await fetch(dataURL)).blob();
  const formData = new FormData();
  formData.append("file", blob, `${name}.png`);
  formData.append("name", name);

  try {
    const response = await fetch("/api/register", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (!response.ok) throw new Error("註冊失敗: " + result.message);
    alert(result.message);
    nameInput.value = null;

    if (result.bbox) {
      const { x1, y1, x2, y2 } = result.bbox;

      const img = new Image();
      img.onload = () => {
        canvasPreview.width = img.width;
        canvasPreview.height = img.height;
        ctxPreview.clearRect(0, 0, canvasPreview.width, canvasPreview.height);
        ctxPreview.drawImage(img, 0, 0);

        ctxPreview.strokeStyle = "lime";
        ctxPreview.lineWidth = 2;
        ctxPreview.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctxPreview.fillStyle = "lime";
        ctxPreview.font = "16px Arial";
        ctxPreview.fillText(name, x1, y1 - 10);
      };
      img.src = dataURL;
    }
  } catch (err) {
    alert(err.message);
  }
  registerBtn.disabled = false;
  registerBtn.style.cursor = "pointer";
}

registerBtn.addEventListener("click", register);

nameInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    event.preventDefault();
    register();
  }
});

const modeToggleInput = document.getElementById("modeToggleInput");

modeToggleInput.addEventListener("change", () => {
  useCamera = !modeToggleInput.checked;

  ctxPreview.clearRect(0, 0, canvasPreview.width, canvasPreview.height);
  canvasPreview.style.display = "none";
  hasPreviewImage = false;

  if (useCamera) {
    canvas.style.display = "block";
    takePhotoBtn.style.display = "inline-block";
    customFileBtn.style.display = "none";
    initSelfieSegmentation();
  } else {
    canvas.style.display = "none";
    takePhotoBtn.style.display = "none";
    customFileBtn.style.display = "inline-block";
    stopSegmentation();
  }
});

takePhotoBtn.addEventListener("click", () => {
  const dataURL = canvas.toDataURL("image/png");
  const img = new Image();
  img.onload = () => {
    canvasPreview.width = img.width;
    canvasPreview.height = img.height;
    ctxPreview.clearRect(0, 0, canvasPreview.width, canvasPreview.height);
    ctxPreview.drawImage(img, 0, 0);
    canvasPreview.style.display = "block";
    hasPreviewImage = true;
  };
  img.src = dataURL;
});

document.getElementById("customFileBtn").addEventListener("click", () => {
  document.getElementById("fileInput").click();
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      canvasPreview.width = 360;
      canvasPreview.height = 270;
      ctxPreview.clearRect(0, 0, canvasPreview.width, canvasPreview.height);

      const scale = Math.min(
        canvasPreview.width / img.width,
        canvasPreview.height / img.height
      );
      const imgWidth = img.width * scale;
      const imgHeight = img.height * scale;
      const offsetX = (canvasPreview.width - imgWidth) / 2;
      const offsetY = (canvasPreview.height - imgHeight) / 2;

      ctxPreview.drawImage(img, offsetX, offsetY, imgWidth, imgHeight);
      canvasPreview.style.display = "block";
      hasPreviewImage = true;
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

goRecognizeBtn.addEventListener("click", () => {
  window.location.href = "/recognize";
});
