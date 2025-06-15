const registerBtn = document.getElementById("registerBtn");
const recognizeBtn = document.getElementById("recognizeBtn");

registerBtn.addEventListener("click", () => {
  window.location.href = "/register";
});

recognizeBtn.addEventListener("click", () => {
  window.location.href = "/recognize";
});
