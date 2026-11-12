/* =========================================================
   Fichário — Assistente de Documentos com RAG
   Pipeline: PDF -> texto -> chunks -> embeddings -> busca por
   similaridade de cosseno -> resposta da LLM com contexto.
   Tudo roda no navegador. Só a chave de API do Gemini é externa.
   ========================================================= */

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

const EMBED_MODEL = "gemini-embedding-001";
const CHAT_MODEL  = "gemini-3.6-flash";
const CHUNK_SIZE  = 900;      // caracteres por ficha
const CHUNK_OVERLAP = 150;    // sobreposição entre fichas
const TOP_K = 4;              // fichas usadas por pergunta
const OCR_LANG = "por";       // idioma do Tesseract.js
const OCR_SCALE = 2;          // resolução do render da página para o OCR
const IMAGE_TYPES = ["image/jpeg", "image/png"];
const THUMB_WIDTH = 240;      // px da miniatura (exibida a 120px; 2x para telas densas)
const PREVIEW_WIDTH = 1400;   // px máximos da imagem ampliada
const JPEG_QUALITY = 0.72;

const state = {
  apiKey: localStorage.getItem("fichario_api_key") || "",
  chunks: [],       // { id, text, page, ocr, image: { thumb, full } | null, embedding }
  ready: false
};

/* ---------- elementos ---------- */
const el = {
  btnConfig: document.getElementById("btn-config"),
  modal: document.getElementById("config-modal"),
  apiKeyInput: document.getElementById("api-key-input"),
  btnSaveKey: document.getElementById("btn-save-key"),
  btnCloseModal: document.getElementById("btn-close-modal"),

  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  docStatus: document.getElementById("doc-status"),
  docName: document.getElementById("doc-name"),
  docMeta: document.getElementById("doc-meta"),
  progressFill: document.getElementById("progress-fill"),
  docStage: document.getElementById("doc-stage"),
  cardIndex: document.getElementById("card-index"),

  imageModal: document.getElementById("image-modal"),
  imageModalImg: document.getElementById("image-modal-img"),
  imageModalCaption: document.getElementById("image-modal-caption"),
  btnCloseImage: document.getElementById("btn-close-image"),

  chatLog: document.getElementById("chat-log"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
};

/* ---------- configuração da API key ---------- */
el.btnConfig.addEventListener("click", () => {
  el.apiKeyInput.value = state.apiKey;
  el.modal.classList.remove("hidden");
});
el.btnCloseModal.addEventListener("click", () => el.modal.classList.add("hidden"));
el.btnSaveKey.addEventListener("click", () => {
  state.apiKey = el.apiKeyInput.value.trim();
  localStorage.setItem("fichario_api_key", state.apiKey);
  el.modal.classList.add("hidden");
});

function requireApiKey() {
  if (!state.apiKey) {
    el.modal.classList.remove("hidden");
    return false;
  }
  return true;
}

/* ---------- upload / dropzone ---------- */
el.dropzone.addEventListener("click", () => el.fileInput.click());
el.dropzone.addEventListener("dragover", (e) => { e.preventDefault(); el.dropzone.classList.add("drag"); });
el.dropzone.addEventListener("dragleave", () => el.dropzone.classList.remove("drag"));
el.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  el.dropzone.classList.remove("drag");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
el.fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  const isImage = IMAGE_TYPES.includes(file.type);
  if (file.type !== "application/pdf" && !isImage) {
    alert("Por enquanto o Fichário só lê arquivos PDF, JPG ou PNG.");
    return;
  }
  if (!requireApiKey()) return;

  resetDocState();
  el.docStatus.classList.remove("hidden");
  el.docName.textContent = file.name;
  el.docMeta.textContent = `${(file.size / 1024).toFixed(0)} KB`;

  try {
    let pages;
    if (isImage) {
      setStage("Lendo imagem com OCR…", 8);
      pages = await extractImageText(file);
    } else {
      setStage("Extraindo texto do PDF…", 8);
      pages = await extractPdfText(file, (page, total) => {
        const pct = 8 + Math.round(((page - 1) / total) * 14);
        setStage(`Lendo página ${page}/${total} com OCR…`, pct);
      });
    }

    setStage("Recortando em fichas…", 22);
    const chunks = chunkPages(pages);

    setStage(`Gerando embeddings (0/${chunks.length})…`, 30);
    await embedChunks(chunks, (done, total) => {
      const pct = 30 + Math.round((done / total) * 65);
      setStage(`Gerando embeddings (${done}/${total})…`, pct);
    });

    state.chunks = chunks;
    state.ready = true;

    setStage("Catalogado. Pronto para perguntas.", 100);
    renderCardIndex(chunks);
    enableChat();
  } catch (err) {
    console.error(err);
    setStage(`Erro: ${err.message}`, 0);
  }
}

function resetDocState() {
  state.chunks = [];
  state.ready = false;
  el.cardIndex.innerHTML = `<p class="empty-note">Processando documento…</p>`;
  disableChat();
}

function setStage(text, pct) {
  el.docStage.textContent = text;
  el.progressFill.style.width = pct + "%";
}

/* ---------- extração de texto do PDF ---------- */
// Páginas com texto são lidas pelo PDF.js; páginas sem texto (scans) são
// renderizadas num canvas e lidas com OCR local (Tesseract.js).
async function extractPdfText(file, onOcr) {
  const buffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
  const pages = [];
  let worker = null;
  try {
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      let text = content.items.map((it) => it.str).join(" ").replace(/\s+/g, " ").trim();
      let ocr = false;
      let image = null;
      if (!text) {
        onOcr(i, pdf.numPages);
        worker = worker || await Tesseract.createWorker(OCR_LANG);
        const viewport = page.getViewport({ scale: OCR_SCALE });
        const canvas = document.createElement("canvas");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
        text = await ocrImage(worker, canvas);
        ocr = true;
        image = makePageImages(canvas);
      }
      if (text) pages.push({ page: i, text, ocr, image });
    }
  } finally {
    if (worker) await worker.terminate();
  }
  if (!pages.length) throw new Error("Não consegui extrair texto deste PDF, nem com OCR.");
  return pages;
}

/* ---------- OCR de imagens (JPG/PNG) ---------- */
async function extractImageText(file) {
  const worker = await Tesseract.createWorker(OCR_LANG);
  try {
    const text = await ocrImage(worker, file);
    if (!text) throw new Error("Não encontrei texto nesta imagem.");
    const bitmap = await createImageBitmap(file);
    const image = makePageImages(bitmap);
    bitmap.close();
    return [{ page: 1, text, ocr: true, image }];
  } finally {
    await worker.terminate();
  }
}

async function ocrImage(worker, source) {
  const { data } = await worker.recognize(source);
  return data.text.replace(/\s+/g, " ").trim();
}

/* ---------- miniaturas das páginas de OCR (só em memória) ---------- */
function makePageImages(source) {
  return {
    thumb: toJpeg(source, THUMB_WIDTH),
    full: toJpeg(source, PREVIEW_WIDTH)
  };
}

function toJpeg(source, maxWidth) {
  const scale = Math.min(1, maxWidth / source.width);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(source.width * scale);
  canvas.height = Math.round(source.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff"; // fundo branco para PNG com transparência
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

/* ---------- chunking com sobreposição ---------- */
function chunkPages(pages) {
  const chunks = [];
  let id = 0;
  for (const { page, text, ocr, image } of pages) {
    let start = 0;
    while (start < text.length) {
      const end = Math.min(start + CHUNK_SIZE, text.length);
      const slice = text.slice(start, end).trim();
      if (slice) chunks.push({ id: id++, page, text: slice, ocr, image: image || null, embedding: null });
      if (end === text.length) break;
      start = end - CHUNK_OVERLAP;
    }
  }
  return chunks;
}

/* ---------- embeddings (API Gemini) ---------- */
async function embedText(text) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent?key=${state.apiKey}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: `models/${EMBED_MODEL}`,
      content: { parts: [{ text }] }
    })
  });
  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`Falha ao gerar embedding (HTTP ${res.status}): ${errBody.slice(0, 200)}`);
  }
  const data = await res.json();
  return data.embedding.values;
}

async function embedChunks(chunks, onProgress) {
  for (let i = 0; i < chunks.length; i++) {
    chunks[i].embedding = await embedText(chunks[i].text);
    onProgress(i + 1, chunks.length);
  }
}

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function topMatches(queryEmbedding, k) {
  return state.chunks
    .map((c) => ({ ...c, score: cosineSimilarity(queryEmbedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

/* ---------- índice visual de fichas ---------- */
function renderCardIndex(chunks) {
  el.cardIndex.innerHTML = "";
  const shown = chunks.slice(0, 40); // limite visual, o índice de busca usa todas
  for (const c of shown) {
    const div = document.createElement("div");
    div.className = "index-card";
    div.innerHTML = `<b>Ficha ${String(c.id + 1).padStart(3, "0")} · pág. ${c.page}${c.ocr ? " · OCR" : ""}</b>${escapeHtml(c.text.slice(0, 90))}…`;
    if (c.image) {
      div.classList.add("has-thumb");
      const thumb = document.createElement("button");
      thumb.type = "button";
      thumb.className = "thumb";
      thumb.title = "Ampliar página";
      thumb.innerHTML = `<img src="${c.image.thumb}" alt="Miniatura da pág. ${c.page}">`;
      thumb.addEventListener("click", () => openImageModal(c.image.full, `Pág. ${c.page} · OCR`));
      div.prepend(thumb);
    }
    el.cardIndex.appendChild(div);
  }
  if (chunks.length > shown.length) {
    const note = document.createElement("p");
    note.className = "empty-note";
    note.textContent = `+ ${chunks.length - shown.length} fichas adicionais indexadas.`;
    el.cardIndex.appendChild(note);
  }
}

/* ---------- modal da imagem ampliada ---------- */
function openImageModal(src, caption) {
  el.imageModalImg.src = src;
  el.imageModalCaption.textContent = caption;
  el.imageModal.classList.remove("hidden");
  el.btnCloseImage.focus();
}
function closeImageModal() {
  el.imageModal.classList.add("hidden");
  el.imageModalImg.removeAttribute("src");
}
el.btnCloseImage.addEventListener("click", closeImageModal);
el.imageModal.addEventListener("click", (e) => {
  if (e.target === el.imageModal) closeImageModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !el.imageModal.classList.contains("hidden")) closeImageModal();
});

/* ---------- chat ---------- */
function enableChat() {
  el.chatInput.disabled = false;
  el.chatSend.disabled = false;
  el.chatInput.focus();
}
function disableChat() {
  el.chatInput.disabled = true;
  el.chatSend.disabled = true;
}

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.chatInput.value.trim();
  if (!question || !state.ready) return;
  if (!requireApiKey()) return;

  addMessage("user", question);
  el.chatInput.value = "";
  disableChat();
  const loadingMsg = addLoadingMessage();

  try {
    const qEmbedding = await embedText(question);
    const matches = topMatches(qEmbedding, TOP_K);
    const answer = await askLLM(question, matches);
    loadingMsg.remove();
    addMessage("assistant", answer, matches);
  } catch (err) {
    console.error(err);
    loadingMsg.remove();
    addMessage("assistant", `Não consegui responder: ${err.message}`);
  } finally {
    enableChat();
    el.chatInput.focus();
  }
});

async function askLLM(question, matches) {
  const context = matches
    .map((m) => `[Ficha ${m.id + 1} — pág. ${m.page}${m.ocr ? " — OCR" : ""}]\n${m.text}`)
    .join("\n\n---\n\n");

  const ocrNote = matches.some((m) => m.ocr)
    ? `Trechos marcados com OCR foram lidos de imagem por reconhecimento óptico e podem conter erros de leitura ` +
      `(letras ou números trocados, palavras quebradas). Leve isso em conta e, se um dado desses trechos parecer duvidoso, avise o usuário. `
    : "";

  const prompt =
    `Você é um assistente que responde exclusivamente com base nos trechos de documento fornecidos abaixo. ` +
    `Se a resposta não estiver nos trechos, diga claramente que o documento não traz essa informação — não invente. ` +
    ocrNote +
    `Responda em português, de forma direta.\n\n` +
    `TRECHOS DISPONÍVEIS:\n${context}\n\n` +
    `PERGUNTA: ${question}`;

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${CHAT_MODEL}:generateContent?key=${state.apiKey}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }]
    })
  });
  if (!res.ok) {
    const errBody = await res.text();
    throw new Error(`HTTP ${res.status}: ${errBody.slice(0, 200)}`);
  }
  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "(resposta vazia)";
}

function addMessage(role, text, sources) {
  const wrap = document.createElement("div");
  wrap.className = `chat-msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const srcWrap = document.createElement("div");
    srcWrap.className = "sources";
    for (const s of sources) {
      const tag = document.createElement("span");
      tag.className = "source-tag";
      tag.title = s.text.slice(0, 160) + "…";
      tag.textContent = `Ficha ${s.id + 1} · pág. ${s.page}${s.ocr ? " · OCR" : ""} · ${(s.score * 100).toFixed(0)}%`;
      srcWrap.appendChild(tag);
    }
    wrap.appendChild(srcWrap);
  }

  el.chatLog.appendChild(wrap);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
  return wrap;
}

function addLoadingMessage() {
  const wrap = document.createElement("div");
  wrap.className = "chat-msg loading";
  wrap.innerHTML = `<div class="bubble">Consultando as fichas…</div>`;
  el.chatLog.appendChild(wrap);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
  return wrap;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
