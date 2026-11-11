/* Keep the project-page selector in sync with the standalone K3D gallery. */
const viewer = document.querySelector("#qualitative-viewer");
const noiseButtons = document.querySelectorAll("[data-noise]");
const noiseStatus = document.querySelector("#noise-status");
let selectedNoise = "0";

function noiseLabel(noise) {
  return noise === "0" ? "Clean observations" : `${noise}% RMS Gaussian noise`;
}

noiseButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectedNoise = button.dataset.noise;
    noiseButtons.forEach((other) => {
      const selected = other === button;
      other.classList.toggle("selected", selected);
      other.setAttribute("aria-pressed", String(selected));
    });
    noiseStatus.textContent = `Loading ${noiseLabel(selectedNoise).toLowerCase()}…`;
    viewer.contentWindow.postMessage({ type: "fast-set-noise", noise: selectedNoise }, "*");
  });
});

window.addEventListener("message", (event) => {
  if (event.source !== viewer.contentWindow) return;
  const message = event.data;
  if (message?.type === "fast-viewer-height" && Number.isFinite(message.height)) {
    viewer.style.height = `${Math.min(Math.max(message.height, 400), 2200)}px`;
  } else if (message?.type === "fast-viewer-expand") {
    viewer.classList.toggle("viewer-expanded", message.active === true);
    document.body.style.overflow = message.active === true ? "hidden" : "";
  } else if (message?.type === "fast-viewer-ready") {
    if (message.noise === selectedNoise) noiseStatus.textContent = noiseLabel(selectedNoise);
    else viewer.contentWindow.postMessage({ type: "fast-set-noise", noise: selectedNoise }, "*");
  } else if (message?.type === "fast-viewer-error") {
    noiseStatus.textContent = "3D views require WebGL. The original figure is linked below.";
  }
});

/* Clipboard API on localhost/HTTPS, with a fallback for direct file opening. */
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Direct file previews may expose the API without clipboard permission.
    }
  }
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.append(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
  if (!copied) throw new Error("Clipboard permission is unavailable.");
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const label = button.querySelector("span");
    const previous = label.textContent;
    try {
      await copyText(document.getElementById(button.dataset.copy).textContent);
      label.textContent = "Copied!";
      document.querySelector("#copy-status").textContent = "Copied to clipboard.";
    } catch {
      label.textContent = "Select text to copy";
      document.querySelector("#copy-status").textContent = "Please select the text and copy it manually.";
    }
    window.setTimeout(() => { label.textContent = previous; }, 2200);
  });
});
