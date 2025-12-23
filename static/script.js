async function uploadImage() {
  const fileInput = document.getElementById("imageInput");
  const preview = document.getElementById("preview");
  const caption = document.getElementById("caption");

  if (!fileInput.files.length) {
    alert("Please upload an image!");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("image", file);

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = "block";
  };
  reader.readAsDataURL(file);

  caption.textContent = "Processing...";

  const response = await fetch("/api/detect", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (data.caption) {
    caption.textContent = data.caption;
  } else {
    caption.textContent = "No objects detected.";
  }
}
