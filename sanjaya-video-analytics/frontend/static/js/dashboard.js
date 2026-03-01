async function listVideos() {
  const r = await fetch("/api/videos"); const j = await r.json();
  const sel = document.getElementById("dashVideo");
  sel.innerHTML = '<option value="">Select video…</option>';
  (j.videos || []).forEach(v => { const o = document.createElement("option"); o.value=v; o.textContent=v; sel.appendChild(o); });
}

function drawSkeleton(canvasId){
  const c = document.getElementById(canvasId); const ctx = c.getContext("2d");
  c.width = 480; c.height = 270; ctx.fillStyle="#0d1326"; ctx.fillRect(0,0,c.width,c.height);
  ctx.fillStyle="#1b2745"; ctx.fillRect(8,8,c.width-16,c.height-16);
  ctx.fillStyle="#2b3c6b"; ctx.fillRect(24,24,180,100);
  ctx.fillRect(220,60,120,60); ctx.fillRect(360,120,100,80);
}

function drawFrame(canvasId, imgUrl, objects=[], persons=[]) {
  if(!imgUrl){ drawSkeleton(canvasId); return; }
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width; canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 2;
    persons.forEach(p => {
      const [x,y,w,h] = p.bbox || [0,0,0,0];
      ctx.strokeStyle = "#00e5ff"; ctx.strokeRect(x,y,w,h);
      ctx.fillStyle = "rgba(0,229,255,0.15)"; ctx.fillRect(x,y,w,h);
      ctx.fillStyle = "#00e5ff"; ctx.font = "14px sans-serif";
      ctx.fillText(`${p.id}${p.posture?' • '+p.posture:''}`, x+4, y+16);
    });
    objects.forEach(o => {
      const [x,y,w,h] = o.bbox || [0,0,0,0];
      ctx.strokeStyle = "#ffb74d"; ctx.strokeRect(x,y,w,h);
      ctx.fillStyle = "rgba(255,183,77,0.15)"; ctx.fillRect(x,y,w,h);
      ctx.fillStyle = "#ffc36e"; ctx.font = "14px sans-serif";
      ctx.fillText(`${o.class}`, x+4, y+16);
    });
  };
  img.src = imgUrl;
}

async function loadStats(video) {
  const r = await fetch(`/api/stats?video=${encodeURIComponent(video)}`); const j = await r.json();
  const t = j.totals || {};
  document.getElementById("kpiPersons").textContent = t.persons ?? 0;
  document.getElementById("kpiObjects").textContent = t.objects ?? 0;
  document.getElementById("kpiObjClasses").textContent = t.object_classes ?? 0;
  document.getElementById("kpiActivities").textContent = t.activities ?? 0;
  document.getElementById("kpiFrames").textContent = t.frames ?? 0;
  const top = (j.top_objects || [])[0]; document.getElementById("kpiTopObject").textContent = top ? `${top.class}` : "–";
  const list = document.getElementById("topObjects"); list.innerHTML = "";
  (j.top_objects || []).forEach(o=>{
    const li = document.createElement("li"); li.className = "list-group-item d-flex justify-content-between align-items-center";
    li.innerHTML = `<span><i class="bi bi-box-seam me-2"></i>${o.class}</span><span class="badge bg-primary-soft">${o.count}</span>`;
    list.appendChild(li);
  });
  const chips = document.getElementById("activityChips"); chips.innerHTML = "";
  (j.activities || []).forEach(a=>{
    const c = document.createElement("span"); c.className="chip"; c.textContent=a; chips.appendChild(c);
  });
}

async function loadFrames(video) {
  ["frameCanvas0","frameCanvas1","frameCanvas2"].forEach(drawSkeleton);
  const r = await fetch(`/api/frames?video=${encodeURIComponent(video)}`); const j = await r.json();
  const frames = j.frames || [];
  for (let i=0;i<3;i++) {
    const f = frames[i];
    if (f) drawFrame(`frameCanvas${i}`, f.image, f.objects, f.persons);
  }
}

function drawHeatmapGrid(ctx, w, h) {
  const cw = Math.floor(w/3), ch = Math.floor(h/3);
  ctx.strokeStyle = "#233058"; ctx.lineWidth = 1;
  for (let i=1;i<3;i++){ ctx.beginPath(); ctx.moveTo(cw*i,0); ctx.lineTo(cw*i,h); ctx.stroke(); }
  for (let i=1;i<3;i++){ ctx.beginPath(); ctx.moveTo(0,ch*i); ctx.lineTo(w,ch*i); ctx.stroke(); }
}

async function loadHeatmap(video) {
  const canvas = document.getElementById("heatmapCanvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 600; canvas.height = 360;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  drawHeatmapGrid(ctx, canvas.width, canvas.height);
  const r = await fetch(`/api/heatmap?video=${encodeURIComponent(video)}`); const j = await r.json();
  const zones = j.zones || {};
  const mapIndex = {"Z1":[0,0],"Z2":[1,0],"Z3":[2,0],"Z4":[0,1],"Z5":[1,1],"Z6":[2,1],"Z7":[0,2],"Z8":[1,2],"Z9":[2,2]};
  const cw = canvas.width/3, ch = canvas.height/3;
  Object.entries(zones).forEach(([z,c])=>{
    const pos = mapIndex[z] || [1,1];
    const x = pos[0]*cw, y = pos[1]*ch;
    const intensity = Math.min(1, c/10);
    ctx.fillStyle = `rgba(110,168,254,${0.15+0.55*intensity})`;
    ctx.fillRect(x+1, y+1, cw-2, ch-2);
    ctx.fillStyle = "#cbd6f7"; ctx.font="16px sans-serif";
    ctx.fillText(`${z}: ${c}`, x+10, y+24);
  });
}

async function loadGraph(video){
  const r = await fetch(`/api/graph?video=${encodeURIComponent(video)}`); const j = await r.json();
  window.renderGraph(j.nodes||[], j.edges||[]);
}

function typewriter(el, text){
  el.textContent=""; let i=0;
  const id = setInterval(()=>{ el.textContent += text[i++] || ""; if(i>=text.length) clearInterval(id); }, 15);
}

async function askRag(video, q, mode){
  const thinking = document.getElementById("ragThinking");
  thinking.textContent = "Thinking…";
  const r = await fetch("/rag/ask", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({video_name: video, question: q, mode})
  });
  const j = await r.json();
  thinking.textContent = "";
  const ansEl = document.getElementById("ragAnswer");
  const evEl = document.getElementById("ragEvidence");
  typewriter(ansEl, `Answer: ${j.answer || "N/A"}`);
  const insights = (j.insights||[]).map(i=>`- ${i}`).join("<br>");
  evEl.innerHTML = `<div class="mt-2"><b>Insights</b><br>${insights||"N/A"}</div>`;
  const ev = (j.evidence||[]).map(e=>`<div class="ev"><div class="ev-head">${e.type||'json'} ${e.path?('('+e.path+')'):''}</div><pre>${(e.snippet||'').slice(0,300)}</pre></div>`).join("");
  evEl.innerHTML += `<div class="mt-2"><b>Evidence</b>${ev||"N/A"}</div>`;
}

function hookPromptChips(){
  const chips = document.getElementById("promptChips");
  chips.querySelectorAll(".chip").forEach(c=>{
    c.addEventListener("click", ()=>{ document.getElementById("ragQ").value = c.dataset.prompt || c.textContent; });
  });
}

function exportReport(video){
  fetch(`/api/stats?video=${encodeURIComponent(video)}`).then(r=>r.json()).then(j=>{
    const blob = new Blob([JSON.stringify(j,null,2)], {type:"application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href=url; a.download=`${video}_report.json`; a.click(); URL.revokeObjectURL(url);
  });
}

document.addEventListener("DOMContentLoaded", async ()=>{
  await listVideos(); hookPromptChips();
  const sel = document.getElementById("dashVideo");
  sel.addEventListener("change", async ()=>{
    const v = sel.value; if(!v) return;
    await loadStats(v); await loadFrames(v); await loadHeatmap(v); await loadGraph(v);
  });
  document.getElementById("btnRefresh").addEventListener("click", async ()=>{
    const v = sel.value; if(!v) return;
    await loadStats(v); await loadFrames(v); await loadHeatmap(v); await loadGraph(v);
  });
  document.getElementById("btnExport").addEventListener("click", ()=>{ const v = sel.value; if(v) exportReport(v); });
  document.getElementById("ragGo").addEventListener("click", async ()=>{
    const v = sel.value; const q = document.getElementById("ragQ").value; const m = document.getElementById("ragMode").value;
    if(!v || !q) return;
    await askRag(v,q,m);
  });
});