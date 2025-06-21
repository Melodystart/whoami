const video = document.createElement("video");
video.style.display = "none";
document.body.appendChild(video);

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const registerBtn = document.getElementById("registerBtn");
const fileInput = document.getElementById("fileInput");
const customFileBtn = document.getElementById("customFileBtn");
const takePhotoBtn = document.getElementById("takePhotoBtn");
const previewImg = document.getElementById("preview");
const modeToggleBtn = document.getElementById("modeToggleBtn");
const nameInput = document.getElementById("nameInput");
const goRecognizeBtn = document.getElementById("goRecognizeBtn");

let useCamera = true;
let selfieSegmentation;
let camera;

navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
    initSelfieSegmentation();
  })
  .catch((err) => {
    console.error("無法取得攝影機:", err);
    alert("攝影機無法使用，請開啟鏡頭或切換為圖片上傳");
  });

function initSelfieSegmentation() {
  const selfieSegmentation = new SelfieSegmentation({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
  });

  selfieSegmentation.setOptions({ modelSelection: 1 });
  selfieSegmentation.onResults(onResults);

  const camera = new Camera(video, {
    onFrame: async () => {
      await selfieSegmentation.send({ image: video });
    },
    width: 360,
    height: 270,
  });

  camera.start();
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

  if (!previewImg.src) {
    alert("請先拍照或上傳圖片");
    registerBtn.disabled = false;
    registerBtn.style.cursor = "pointer";
    return;
  }

  const img = new Image();
  img.crossOrigin = "anonymous"; // 若 src 為 base64 可省略，否則避免跨域問題
  img.src = previewImg.src;

  // 當圖片載入完成後，畫到 canvas 上
  img.onload = async () => {
    // 設定 canvas 尺寸與圖片一致
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // 清空 canvas 並畫出圖片
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    const dataURL = canvas.toDataURL("image/png");
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
    } catch (err) {
      alert(err.message);
    }
    registerBtn.disabled = false;
    registerBtn.style.cursor = "pointer";
  };
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

  previewImg.src = "";
  previewImg.style.display = "none";

  if (useCamera) {
    canvas.style.display = "block";
    takePhotoBtn.style.display = "inline-block";
    customFileBtn.style.display = "none";
    initSelfieSegmentation();
  } else {
    canvas.style.display = "none";
    takePhotoBtn.style.display = "none";
    customFileBtn.style.display = "inline-block";
    if (camera) camera.stop();
  }
});

takePhotoBtn.addEventListener("click", () => {
  const dataURL = canvas.toDataURL("image/png");
  previewImg.src = dataURL;
  previewImg.style.display = "block";
});

document.getElementById("customFileBtn").addEventListener("click", () => {
  document.getElementById("fileInput").click();
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    previewImg.src = event.target.result;
    previewImg.style.display = "block";
  };
  reader.readAsDataURL(file);
});

goRecognizeBtn.addEventListener("click", () => {
  window.location.href = "/recognize";
});
