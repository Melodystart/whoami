const video = document.createElement("video");
video.style.display = "none";
document.body.appendChild(video);

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const captureBtn = document.getElementById("captureBtn");

navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
    initSelfieSegmentation();
  })
  .catch((err) => console.error("無法取得攝影機:", err));

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
    width: 640,
    height: 480,
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
  captureBtn.disabled = true;
  captureBtn.style.cursor = "not-allowed";
  const name = document.getElementById("nameInput").value.trim();
  if (!name) {
    alert("請輸入姓名");
    return;
  }

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

    if (!response.ok) throw new Error("註冊失敗");

    const result = await response.json();
    console.log("註冊成功:", result);
    alert(result.message);
  } catch (err) {
    console.error("註冊失敗:", err);
    alert("註冊失敗");
  }
  captureBtn.disabled = false;
  captureBtn.style.cursor = "pointer";
}

captureBtn.addEventListener("click", register);

const nameInput = document.getElementById("nameInput");
nameInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    event.preventDefault();
    register();
  }
});
