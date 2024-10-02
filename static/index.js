function initializeFileDrop() {
  const fileWrapper = document.getElementById("file-wrapper");
  const fileInput = document.getElementById("file");
  const fileLabel = document.getElementById("file-label");

  fileWrapper.addEventListener("keydown", (e) => {
    if (e.key === "Enter") fileInput.click();
  });

  fileWrapper.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.stopPropagation();
  });

  fileWrapper.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.files = e.dataTransfer.files;
    fileLabel.textContent = fileInput.files[0].name;
    fileInput.dispatchEvent(new Event("change", { bubbles: true }));
    fileInput.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

function submitForm(e) {
  e.preventDefault();

  const fileInput = document.getElementById("file");
  const urlInput = document.getElementById("url");

  if (fileInput.files.length === 0 && urlInput.value === "") {
    alert("Please select a file or enter a URL");
    return;
  } else if (fileInput.files.length > 0 && urlInput.value !== "") {
    alert("Please select a file or enter a URL, not both");
    return;
  }

  const form = document.getElementById("form");
  form.submit();
}

addEventListener("DOMContentLoaded", initializeFileDrop);
