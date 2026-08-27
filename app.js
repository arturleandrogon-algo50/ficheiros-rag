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

const state = {
  apiKey: localStorage.getItem("fichario_api_key") || "",
  chunks: [],       // { id, text, page, embedding }
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
  if (file.type !== "application/pdf") {
    alert("Por enquanto o Fichário só lê arquivos PDF.");
    return;
  }
  if (!requireApiKey()) return;

  resetDocState();
  el.docStatus.classList.remove("hidden");
  el.docName.textContent = file.name;
  el.docMeta.textContent = `${(file.size / 1024).toFixed(0)} KB`;

  try {
    setStage("Extraindo texto do PDF…", 8);
    const pages = await extractPdfText(file);

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
async function extractPdfText(file) {
  const buffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const text = content.items.map((it) => it.str).join(" ").replace(/\s+/g, " ").trim();
    if (text) pages.push({ page: i, text });
  }
  if (!pages.length) throw new Error("Não consegui extrair texto deste PDF (pode ser um scan sem OCR).");
  return pages;
}

/* ---------- chunking com sobreposição ---------- */
function chunkPages(pages) {
  const chunks = [];
  let id = 0;
  for (const { page, text } of pages) {
    let start = 0;
    while (start < text.length) {
      const end = Math.min(start + CHUNK_SIZE, text.length);
      const slice = text.slice(start, end).trim();
      if (slice) chunks.push({ id: id++, page, text: slice, embedding: null });
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
    div.innerHTML = `<b>Ficha ${String(c.id + 1).padStart(3, "0")} · pág. ${c.page}</b>${escapeHtml(c.text.slice(0, 90))}…`;
    el.cardIndex.appendChild(div);
  }
  if (chunks.length > shown.length) {
    const note = document.createElement("p");
    note.className = "empty-note";
    note.textContent = `+ ${chunks.length - shown.length} fichas adicionais indexadas.`;
    el.cardIndex.appendChild(note);
  }
}

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
    .map((m) => `[Ficha ${m.id + 1} — pág. ${m.page}]\n${m.text}`)
    .join("\n\n---\n\n");

  const prompt =
    `Você é um assistente que responde exclusivamente com base nos trechos de documento fornecidos abaixo. ` +
    `Se a resposta não estiver nos trechos, diga claramente que o documento não traz essa informação — não invente. ` +
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
      tag.textContent = `Ficha ${s.id + 1} · pág. ${s.page} · ${(s.score * 100).toFixed(0)}%`;
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
