const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const scoringModeSelect = document.getElementById("scoring-mode");
const llmOptions = document.getElementById("llm-options");
const llmProviderSelect = document.getElementById("llm-provider");
const llmModelSelect = document.getElementById("llm-model");
const apiKeyField = document.getElementById("api-key-field");
const apiKeyLabel = document.getElementById("api-key-label");
const apiKeyInput = document.getElementById("api-key-input");
const ollamaUrlField = document.getElementById("ollama-url-field");
const llmHint = document.getElementById("llm-hint");
const securityNote = document.getElementById("security-note");

// Modal elements
const howItWorksBtn = document.getElementById("how-it-works-btn");
const pipelineModal = document.getElementById("pipeline-modal");
const modalClose = document.getElementById("modal-close");

// Modal open/close handlers
howItWorksBtn.addEventListener("click", () => {
  pipelineModal.classList.add("active");
  document.body.style.overflow = "hidden";
});

modalClose.addEventListener("click", () => {
  pipelineModal.classList.remove("active");
  document.body.style.overflow = "";
});

pipelineModal.addEventListener("click", (e) => {
  if (e.target === pipelineModal) {
    pipelineModal.classList.remove("active");
    document.body.style.overflow = "";
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && pipelineModal.classList.contains("active")) {
    pipelineModal.classList.remove("active");
    document.body.style.overflow = "";
  }
});

// Model options for each provider
const providerModels = {
  openai: [
    { value: "gpt-4o-mini", label: "gpt-4o-mini (fast, cheap)" },
    { value: "gpt-4o", label: "gpt-4o (best quality)" },
    { value: "gpt-4-turbo", label: "gpt-4-turbo" },
    { value: "gpt-3.5-turbo", label: "gpt-3.5-turbo (fastest)" },
  ],
  gemini: [
    { value: "gemini-2.5-flash", label: "Gemini 2.5 Flash (best value)" },
    { value: "gemini-2.0-flash", label: "Gemini 2.0 Flash (stable)" },
    { value: "gemini-2.0-flash-lite", label: "Gemini 2.0 Flash Lite (fastest)" },
    { value: "gemini-3-flash-preview", label: "Gemini 3 Flash (preview)" },
    { value: "gemini-3-pro-preview", label: "Gemini 3 Pro (preview, best)" },
  ],
  huggingface: [
    { value: "mistralai/Mixtral-8x7B-Instruct-v0.1", label: "Mixtral 8x7B" },
    { value: "meta-llama/Llama-2-70b-chat-hf", label: "Llama 2 70B" },
    { value: "codellama/CodeLlama-34b-Instruct-hf", label: "CodeLlama 34B" },
  ],
  ollama: [
    // Your custom models
    { value: "gpt-oss:20b", label: "GPT-OSS 20B (your model)" },
    { value: "gemma3", label: "Gemma 3 4B (3.3GB)" },
    // Official models under 5GB
    { value: "gemma3:1b", label: "Gemma 3 1B (815MB)" },
    { value: "llama3.2:1b", label: "Llama 3.2 1B (1.3GB)" },
    { value: "llama3.2", label: "Llama 3.2 3B (2.0GB)" },
    { value: "phi4-mini", label: "Phi 4 Mini 3.8B (2.5GB)" },
    { value: "codellama", label: "Code Llama 7B (3.8GB)" },
    { value: "llama2-uncensored", label: "Llama 2 Uncensored 7B (3.8GB)" },
    { value: "mistral", label: "Mistral 7B (4.1GB)" },
    { value: "neural-chat", label: "Neural Chat 7B (4.1GB)" },
    { value: "starling-lm", label: "Starling 7B (4.1GB)" },
    { value: "llava", label: "LLaVA 7B (4.5GB)" },
    { value: "deepseek-r1", label: "DeepSeek-R1 7B (4.7GB)" },
    { value: "llama3.1", label: "Llama 3.1 8B (4.7GB)" },
    { value: "granite3.3", label: "Granite 3.3 8B (4.9GB)" },
    { value: "moondream", label: "Moondream 2 1.4B (829MB)" },
  ],
};

const providerHints = {
  openai: "Uses OpenAI API. Requires an API key from platform.openai.com",
  gemini: "Uses Google Gemini API. Requires an API key from ai.google.dev",
  huggingface: "Uses HuggingFace Inference API. Requires a token from huggingface.co",
  ollama: "Uses local Ollama instance. No API key needed. Start with: ollama serve",
};

// Update model dropdown when provider changes
const updateModelOptions = () => {
  const provider = llmProviderSelect.value;
  const models = providerModels[provider] || [];

  llmModelSelect.innerHTML = "";
  models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.value;
    option.textContent = model.label;
    if (index === 0) option.selected = true;
    llmModelSelect.appendChild(option);
  });

  // Update API key field visibility and label
  if (provider === "ollama") {
    apiKeyField.style.display = "none";
    ollamaUrlField.style.display = "block";
    securityNote.style.display = "none"; // No API key needed for Ollama
  } else {
    apiKeyField.style.display = "block";
    ollamaUrlField.style.display = "none";
    securityNote.style.display = "flex"; // Show security note when API key is needed

    if (provider === "openai") {
      apiKeyLabel.textContent = "OpenAI API Key";
      apiKeyInput.placeholder = "sk-...";
    } else if (provider === "gemini") {
      apiKeyLabel.textContent = "Gemini API Key";
      apiKeyInput.placeholder = "AIza...";
    } else if (provider === "huggingface") {
      apiKeyLabel.textContent = "HuggingFace Token";
      apiKeyInput.placeholder = "hf_...";
    }
  }

  // Update hint text
  llmHint.textContent = providerHints[provider] || "";
};

// Show/hide LLM options based on scoring mode
const updateLLMOptionsVisibility = () => {
  if (scoringModeSelect.value === "llm") {
    llmOptions.style.display = "block";
    updateModelOptions();
  } else {
    llmOptions.style.display = "none";
  }
};

scoringModeSelect.addEventListener("change", updateLLMOptionsVisibility);
llmProviderSelect.addEventListener("change", updateModelOptions);
updateLLMOptionsVisibility();

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
    const duration = Math.round(clip.end - clip.start);
    header.textContent = `clip_${index + 1} [${formatTime(clip.start)} - ${formatTime(clip.end)}] ${duration}s`;

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
      link.textContent = "Download";
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
  setStatus("// Processing video... Transcribing and analyzing");
  resultsEl.innerHTML = "";

  const formData = new FormData(form);

  try {
    const response = await fetch("/api/clip-suggestions", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      setStatus("// Error: " + (payload.detail || "Failed to generate clips"));
      return;
    }

    const clipCount = (payload.clips || []).length;
    if (payload.warning) {
      setStatus(`// Complete: ${clipCount} clips found | Warning: ${payload.warning}`);
    } else {
      setStatus(`// Complete: ${clipCount} clips extracted successfully`);
    }

    renderResults(payload.clips || []);
  } catch (error) {
    setStatus("// Error: Connection failed");
  }
});
