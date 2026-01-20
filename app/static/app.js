const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

const setStatus = (message) => {
  statusEl.textContent = message;
};

const renderResults = (clips) => {
  resultsEl.innerHTML = "";
  if (!clips.length) {
    resultsEl.textContent = "No clips found.";
    return;
  }

  clips.forEach((clip, index) => {
    const card = document.createElement("div");
    card.className = "clip";

    const header = document.createElement("div");
    header.className = "clip-header";
    header.textContent = `Clip ${index + 1} (${formatTime(
      clip.start
    )} - ${formatTime(clip.end)})`;

    const scores = document.createElement("div");
    scores.className = "clip-scores";
    const scoreItems = [
      `Combined: ${clip.score}`,
      `Text: ${clip.text_score}`,
      `Coherence: ${clip.coherence_score}`,
      `Distinctiveness: ${clip.distinctiveness_score}`,
    ];
    scores.textContent = scoreItems.join(" | ");

    const text = document.createElement("div");
    text.className = "clip-text";
    text.textContent = clip.text || "No transcript text available.";

    if (clip.file) {
      const link = document.createElement("a");
      link.href = clip.file;
      link.className = "download-btn";
      link.textContent = "Download clip";
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      card.appendChild(header);
      card.appendChild(scores);
      card.appendChild(text);
      card.appendChild(link);
    } else {
      card.appendChild(header);
      card.appendChild(scores);
      card.appendChild(text);
    }

    resultsEl.appendChild(card);
  });
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Uploading and analyzing...");
  resultsEl.innerHTML = "";

  const formData = new FormData(form);

  try {
    const response = await fetch("/api/clip-suggestions", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      setStatus(payload.detail || "Failed to generate clips.");
      return;
    }

    if (payload.warning) {
      setStatus(`Done. Warning: ${payload.warning}`);
    } else {
      setStatus("Done.");
    }

    renderResults(payload.clips || []);
  } catch (error) {
    setStatus("Request failed.");
  }
});
