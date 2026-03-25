const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const resetBtn = document.getElementById("resetBtn");
const attackBtn = document.getElementById("attackBtn");
const epsilonSlider = document.getElementById("epsilonSlider");
const epsilonVal = document.getElementById("epsilonVal");
const loadingBar = document.getElementById("loadingBar");
const resultsCard = document.getElementById("resultsCard");
const btnFGSM = document.getElementById("btnFGSM");
const btnPGD = document.getElementById("btnPGD");
const attackDesc = document.getElementById("attackDesc");

let selectedFile = null;
let selectedAttack = "fgsm";

const attackDescriptions = {
  fgsm: "Fast Gradient Sign Method — single-step attack. Fast and effective.",
  pgd: "Projected Gradient Descent — iterative multi-step attack. Stronger but slower."
};

// ── Upload zone interactions ──
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadZone.style.display = "none";
    previewWrap.style.display = "block";
    attackBtn.disabled = false;
    resultsCard.style.display = "none";
  };
  reader.readAsDataURL(file);
}

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewWrap.style.display = "none";
  uploadZone.style.display = "block";
  attackBtn.disabled = true;
  resultsCard.style.display = "none";
  loadingBar.style.display = "none";
});

// ── Epsilon slider ──
epsilonSlider.addEventListener("input", () => {
  epsilonVal.textContent = parseFloat(epsilonSlider.value).toFixed(3);
});

// ── Attack toggle ──
btnFGSM.addEventListener("click", () => setAttack("fgsm"));
btnPGD.addEventListener("click", () => setAttack("pgd"));

function setAttack(type) {
  selectedAttack = type;
  btnFGSM.classList.toggle("active", type === "fgsm");
  btnPGD.classList.toggle("active", type === "pgd");
  attackDesc.textContent = attackDescriptions[type];
}

// ── Run attack ──
attackBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  loadingBar.style.display = "block";
  resultsCard.style.display = "none";
  attackBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("epsilon", epsilonSlider.value);
  formData.append("attack", selectedAttack);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    renderResults(data);
  } catch (err) {
    alert("Something went wrong. Check the console.");
    console.error(err);
  } finally {
    loadingBar.style.display = "none";
    attackBtn.disabled = false;
  }
});

function renderResults(data) {
  // Status badge
  const badge = document.getElementById("statusBadge");
  if (data.attack_succeeded) {
    badge.textContent = `✓ Attack Succeeded — ${data.attack_type} · ε=${data.epsilon}`;
    badge.className = "status-badge success";
  } else {
    badge.textContent = `✗ Attack Failed — Model held firm · ε=${data.epsilon}`;
    badge.className = "status-badge fail";
  }

  // Images
  document.getElementById("origImg").src = "data:image/png;base64," + data.original_image;
  document.getElementById("noiseImg").src = "data:image/png;base64," + data.noise_image;
  document.getElementById("pertImg").src = "data:image/png;base64," + data.perturbed_image;

  // Labels under images
  document.getElementById("origPred").textContent = data.original_label;
  document.getElementById("pertPred").textContent = data.perturbed_label;

  // Confidence bars
  renderBars("origBars", data.original_preds, false);
  renderBars("pertBars", data.perturbed_preds, true);

  resultsCard.style.display = "block";
  resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderBars(containerId, preds, isPerturbed) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  const row = document.createElement("div");
  row.className = "conf-bar-row";

  preds.forEach(pred => {
    const item = document.createElement("div");
    item.className = "conf-item";

    const labelRow = document.createElement("div");
    labelRow.className = "conf-label-row";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = truncate(pred.label, 28);

    const pctSpan = document.createElement("span");
    pctSpan.className = "conf-pct";
    pctSpan.textContent = pred.confidence.toFixed(1) + "%";

    labelRow.appendChild(nameSpan);
    labelRow.appendChild(pctSpan);

    const track = document.createElement("div");
    track.className = "conf-track";

    const fill = document.createElement("div");
    fill.className = "conf-fill" + (isPerturbed ? " perturbed" : "");
    fill.style.width = "0%";

    track.appendChild(fill);
    item.appendChild(labelRow);
    item.appendChild(track);
    row.appendChild(item);

    // Animate bar
    setTimeout(() => { fill.style.width = pred.confidence + "%"; }, 80);
  });

  container.appendChild(row);
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}