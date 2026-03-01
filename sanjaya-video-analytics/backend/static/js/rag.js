document.addEventListener("DOMContentLoaded", async () => {
  const sel = document.getElementById("ragVideo");
  try {
    const r = await fetch("/api/videos");
    const j = await r.json();
    (j.videos || []).forEach(v => {
      const opt = document.createElement("option");
      opt.value = v; opt.textContent = v;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.warn("Failed to load videos", e);
  }
});

document.getElementById("ragAskBtn").addEventListener("click", askRag);

async function askRag() {
  const videoSel = document.getElementById('ragVideo') || document.getElementById('dashVideo');
  const video = (videoSel && videoSel.value) || '';
  const q = document.getElementById('ragQuestion').value.trim();
  const mode = document.getElementById('ragMode').value || 'hybrid';
  if (!q) return;

  const thinking = document.getElementById('ragThinking');
  const result = document.getElementById('ragResult');
  thinking.textContent = 'Thinking...';
  result.textContent = '';

  try {
    const res = await fetch('/rag/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, video_name: video, mode })
    });
    const data = await res.json();
    const answer = data.answer || 'No answer';
    const insights = (data.insights && data.insights.length) ? data.insights.join('\n') : 'None';
    const evidence = (data.evidence && data.evidence.length)
      ? data.evidence.map(e => `- [${e.type || 'evidence'}] ${e.snippet || ''}`).join('\n')
      : 'None';
    const confidence = (data.confidence !== undefined) ? data.confidence : 'N/A';

    result.textContent =
`Answer: ${answer}

Insights:
${insights}

Evidence:
${evidence}

Confidence: ${confidence}`;
  } catch (err) {
    result.textContent = `Error: ${err.message}`;
  } finally {
    thinking.textContent = '';
  }
}