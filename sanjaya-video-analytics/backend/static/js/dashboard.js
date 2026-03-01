async function listVideos() {
  const r = await fetch("/api/videos"); const j = await r.json();
  const sel = document.getElementById("dashVideo");
  sel.innerHTML = '<option value="">Select video…</option>';
  (j.videos || []).forEach(v => { const o = document.createElement("option"); o.value=v; o.textContent=v; sel.appendChild(o); });
}
function drawSkeleton(canvasId){
  const c = document.getElementById(canvasId); const ctx = c.getContext("2d");
  c.width = 480; c.height = 270; ctx.fillStyle="#f4f6ff"; ctx.fillRect(0,0,c.width,c.height);
  ctx.fillStyle="#e9ecff"; ctx.fillRect(8,8,c.width-16,c.height-16);
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
      ctx.strokeStyle = "#667eea"; ctx.strokeRect(x,y,w,h);
      ctx.fillStyle = "rgba(102,126,234,0.15)"; ctx.fillRect(x,y,w,h);
      ctx.fillStyle = "#334"; ctx.font = "14px sans-serif";
      ctx.fillText(`${p.id}${p.posture?' • '+p.posture:''}`, x+4, y+16);
    });
    objects.forEach(o => {
      const [x,y,w,h] = o.bbox || [0,0,0,0];
      ctx.strokeStyle = "#ffb74d"; ctx.strokeRect(x,y,w,h);
      ctx.fillStyle = "rgba(255,183,77,0.15)"; ctx.fillRect(x,y,w,h);
      ctx.fillStyle = "#653a8e"; ctx.font = "14px sans-serif";
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
}
async function loadFrames() {
  const canvasIds = ['frameCanvas0','frameCanvas1','frameCanvas2'];
  const res = await fetch('/api/frames');
  const data = await res.json();
  (data.frames || []).slice(0,3).forEach((f, idx) => {
    const c = document.getElementById(canvasIds[idx]);
    if (!c) return;
    const ctx = c.getContext('2d');
    const img = new Image();
    img.onload = () => {
      c.width = img.width; c.height = img.height;
      ctx.drawImage(img, 0, 0, img.width, img.height);
      // Draw objects
      (f.objects || []).forEach(o => {
        const [x,y,w,h] = o.bbox || [];
        ctx.strokeStyle = '#ff9800'; ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#ff9800';
        ctx.font = '14px sans-serif';
        ctx.fillText(o.name || 'obj', x+4, y+16);
      });
      // Draw persons
      (f.persons || []).forEach(p => {
        const [x,y,w,h] = p.bbox || [];
        ctx.strokeStyle = '#00bcd4'; ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#00bcd4';
        ctx.font = '14px sans-serif';
        ctx.fillText(p.posture || 'person', x+4, y+16);
      });
    };
    img.src = f.image;
  });
}
function drawHeatmapGrid(ctx, w, h) {
  const cw = Math.floor(w/3), ch = Math.floor(h/3);
  ctx.strokeStyle = "#ddd"; ctx.lineWidth = 1;
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
    ctx.fillStyle = `rgba(118,75,162,${0.15+0.55*intensity})`;
    ctx.fillRect(x+1, y+1, cw-2, ch-2);
    ctx.fillStyle = "#333"; ctx.font="16px sans-serif";
    ctx.fillText(`${z}: ${c}`, x+10, y+24);
  });
}
async function loadGraph(video){
  const r = await fetch(`/api/graph?video=${encodeURIComponent(video)}`); const j = await r.json();
  window.renderGraph(j.nodes||[], j.edges||[]);
}
function hookPromptChips(){
  const chips = document.getElementById("promptChips");
  if (!chips) return;
  chips.querySelectorAll(".chip").forEach(c=>{
    c.addEventListener("click", ()=>{ document.getElementById("ragQuestion").value = c.dataset.prompt || c.textContent; });
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
    // also set RAG selector if present
    const ragSel = document.getElementById("ragVideo"); if (ragSel) ragSel.value = v;
  });
  document.getElementById("btnRefresh").addEventListener("click", async ()=>{
    const v = sel.value; if(!v) return;
    await loadStats(v); await loadFrames(v); await loadHeatmap(v); await loadGraph(v);
  });
  document.getElementById("btnExport").addEventListener("click", ()=>{ const v = sel.value; if(v) exportReport(v); });
});
async function refreshDash(videoName = '') {
  await Promise.all([
    loadStats(videoName),
    loadFrames(),
    loadHeatmap(),
    loadGraph()
  ]);
  // refresh video dropdowns
  try {
    const res = await fetch('/api/videos');
    const data = await res.json();
    const vids = data.videos || [];
    ['dashVideo','ragVideo'].forEach(id => {
      const sel = document.getElementById(id);
      if (!sel) return;
      const cur = sel.value;
      sel.innerHTML = '<option value="">Select video…</option>' + vids.map(v => `<option value="${v}">${v}</option>`).join('');
      if (videoName) sel.value = videoName;
      else if (cur) sel.value = cur;
    });
  } catch(e){}
}
window.refreshDash = refreshDash;
let queryHistory=[];

// Chart instances
let motionChart, eventChart, zoneChart;

// Auto-load event insights on page load if events.json exists
window.addEventListener('DOMContentLoaded', () => {
  console.log('[DOMContentLoaded] Checking for existing event data...');
  loadEventInsightsFromFile();
});

function loadEventInsightsFromFile() {
  fetch('/json_outputs/events.json')
    .then(response => {
      console.log('[loadEventInsights] Response status:', response.status);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(events => {
      console.log('[loadEventInsights] Loaded', events.length, 'events');
      
      if (!events || events.length === 0) {
        console.warn('[loadEventInsights] No events to display');
        return;
      }
      
      const stats = calculateEventStatistics(events);
      console.log('[loadEventInsights] Calculated stats:', stats);
      
      // Update summary cards
      setText('insightTotalEvents', stats.totalEvents);
      setText('insightTimeRange', `${stats.minTime.toFixed(2)}s - ${stats.maxTime.toFixed(2)}s`);
      setText('insightHighPriority', stats.priorityBreakdown.high);
      setText('insightHighPriorityPct', `${stats.priorityPercentages.high.toFixed(1)}% of total`);
      setText('insightUniqueTracks', stats.uniqueTracks);
      setText('insightAvgSpeed', `Avg: ${stats.avgSpeed.toFixed(1)} px/s`);
      
      // Update motion state distribution
      setText('motionStationary', stats.motionStateBreakdown.STATIONARY);
      setText('motionWalking', stats.motionStateBreakdown.WALKING);
      setText('motionMoving', stats.motionStateBreakdown.MOVING);
      setText('motionLoitering', stats.motionStateBreakdown.LOITERING);
      
      setText('motionStationaryPct', `${stats.motionStatePercentages.STATIONARY.toFixed(1)}%`);
      setText('motionWalkingPct', `${stats.motionStatePercentages.WALKING.toFixed(1)}%`);
      setText('motionMovingPct', `${stats.motionStatePercentages.MOVING.toFixed(1)}%`);
      setText('motionLoiteringPct', `${stats.motionStatePercentages.LOITERING.toFixed(1)}%`);
      
      // Update priority breakdown
      setText('priorityHigh', stats.priorityBreakdown.high);
      setText('priorityMedium', stats.priorityBreakdown.medium);
      setText('priorityLow', stats.priorityBreakdown.low);
      
      // Update speed analysis
      setText('speedAvg', `${stats.avgSpeed.toFixed(1)} px/s`);
      setText('speedMax', `${stats.maxSpeed.toFixed(1)} px/s`);
      setText('speedMin', `${stats.minSpeed.toFixed(1)} px/s`);
      
      // Update zone hotspots
      displayZoneHotspots(stats.zoneActivity);
      
      console.log('[loadEventInsights] ✅ Successfully updated all event insights!');
    })
    .catch(err => {
      console.error('[loadEventInsights] ❌ Error:', err);
    });
}

function showStatus(msg,type='info'){
  const el=document.getElementById('uploadStatus');
  el.textContent=msg; el.classList.add('active');
  el.style.background= type==='error' ? '#e53e3e' :
                      type==='success' ? '#38a169' :
                      '#667eea';
}

function handleFileSelected() {
  const file = document.getElementById('videoFile').files[0];
  if (file) {
    console.log('File selected:', file.name);
    showStatus(`File selected: ${file.name}. Click Analyze to process.`, 'info');
  }
}

async function uploadVideo(){
  console.log('uploadVideo called');
  const file=document.getElementById('videoFile').files[0];
  console.log('Selected file:', file);
  if(!file){showStatus('Please choose a video file','error');return;}
  showStatus('Analyzing video...','info');
  const fd=new FormData(); fd.append('file',file);
  try{
    console.log('Sending request to /pipeline/upload');
    const resp=await fetch('/pipeline/upload',{method:'POST',body:fd});
    console.log('Response received:', resp.status);
    const data=await resp.json();
    console.log('Response data:', data);
    if(!resp.ok) throw new Error(data.error||'Upload failed');
    showStatus(`Done. ${data.salient_frames?.length||0} salient frames found`,'success');
    updateStats(data);
    displaySalientFrames(data.salient_frames||[]);
    displayVLMAnalysis(data.vlm_results||[], data.insights||{});
    displayEvents(data);
    
    // Force reload event insights from fresh JSON file
    setTimeout(() => {
      console.log('[Upload] Forcing event insights reload from file');
      loadEventInsightsFromFile();
    }, 800);
    
    // Render charts with error handling - OPTIONAL, don't break if it fails
    try {
      if (typeof renderCharts === 'function') {
        console.log('Rendering charts...');
        renderCharts(data);
        console.log('Charts rendered successfully');
      } else {
        console.warn('renderCharts function not available - charts disabled');
      }
    } catch (chartError) {
      console.error('Chart rendering error (non-critical):', chartError);
    }
  }catch(e){
    console.error('Upload error:', e);
    showStatus(e.message,'error');
  }
}

function updateStats(d){
  const frames=d.salient_frames||[], vlm=d.vlm_results||[];
  let persons=0,objectsList=[],riskLevels=[],zoneAct={};
  
  frames.forEach(f=>{
    // Count persons
    const personCount = Array.isArray(f.persons) ? f.persons.length : (f.persons || 0);
    persons += personCount;
    
    // Collect unique object classes
    (f.all_objects||[]).forEach(o=>{
      const z=o.zone||'Z?';
      zoneAct[z]=(zoneAct[z]||0)+1;
      const objClass = o.class || o.type || 'object';
      if (!objectsList.includes(objClass)) {
        objectsList.push(objClass);
      }
    });
  });
  
  vlm.forEach(v=>{const r=v.behavioral_assessment?.risk_level; if(r) riskLevels.push(r);});
  const risk = riskLevels.includes('high')?'HIGH':riskLevels.includes('medium')?'MEDIUM':riskLevels.length?'LOW':'N/A';
  
  setText('totalFrames',frames.length);
  setText('totalPersons',persons);
  setText('totalObjects',objectsList.join(', ') || '0');
  setText('totalEvents',d.cv_events||0);
  setText('salientCount',frames.length);
  setText('riskLevel',risk);
  displayZoneActivity(zoneAct);
}

function setText(id,val){const el=document.getElementById(id); if(el) el.textContent=val;}

function displayZoneActivity(zoneAct){
  const c=document.getElementById('zoneHeatmap');
  if(!c) return; // Element doesn't exist
  const zones=Object.keys(zoneAct);
  if(!zones.length){c.classList.add('empty'); c.textContent='No zone data'; return;}
  c.classList.remove('empty');
  const max=Math.max(...Object.values(zoneAct));
  c.innerHTML=zones.map(z=>{
    const t=zoneAct[z]/max;
    const cls=t>.7?'hot':t>.4?'mid':'cool';
    return `<div class="zone ${cls}">${z}: ${zoneAct[z]}</div>`;
  }).join('');
}

function displayEvents(d){
  const c=document.getElementById('eventsTimeline');
  if(!c) return; // Element doesn't exist
  const n=d.cv_events||0;
  if(!n){c.classList.add('empty'); c.textContent='No events detected'; return;}
  c.classList.remove('empty');
  c.innerHTML=[...Array(Math.min(n,10)).keys()].map(i=>`<div class="event">Event ${i+1}</div>`).join('');
  
  // Call the new event insights function - ALWAYS call it
  console.log('[displayEvents] Calling displayEventInsights');
  displayEventInsights(d);
}

function displayEventInsights(d) {
  console.log('[displayEventInsights] Called - delegating to loadEventInsightsFromFile');
  loadEventInsightsFromFile();
}

function calculateEventStatistics(events) {
  if (!events || events.length === 0) {
    return getEmptyStats();
  }
  
  const stats = {
    totalEvents: events.length,
    uniqueTracks: new Set(events.map(e => e.track_id)).size,
    motionStateBreakdown: { STATIONARY: 0, WALKING: 0, MOVING: 0, LOITERING: 0 },
    priorityBreakdown: { high: 0, medium: 0, low: 0 },
    zoneActivity: {},
    speeds: [],
    minTime: Math.min(...events.map(e => e.timestamp)),
    maxTime: Math.max(...events.map(e => e.timestamp))
  };
  
  events.forEach(event => {
    // Count motion states
    const state = event.motion_state || 'STATIONARY';
    stats.motionStateBreakdown[state] = (stats.motionStateBreakdown[state] || 0) + 1;
    
    // Count priority levels
    const priority = event.priority || 'low';
    stats.priorityBreakdown[priority] = (stats.priorityBreakdown[priority] || 0) + 1;
    
    // Track zone activity
    const zone = event.zone || 'Unknown';
    stats.zoneActivity[zone] = (stats.zoneActivity[zone] || 0) + 1;
    
    // Collect speeds
    if (event.speed_px_s !== undefined) {
      stats.speeds.push(event.speed_px_s);
    }
  });
  
  // Calculate percentages
  stats.motionStatePercentages = {
    STATIONARY: (stats.motionStateBreakdown.STATIONARY / stats.totalEvents) * 100,
    WALKING: (stats.motionStateBreakdown.WALKING / stats.totalEvents) * 100,
    MOVING: (stats.motionStateBreakdown.MOVING / stats.totalEvents) * 100,
    LOITERING: (stats.motionStateBreakdown.LOITERING / stats.totalEvents) * 100
  };
  
  stats.priorityPercentages = {
    high: (stats.priorityBreakdown.high / stats.totalEvents) * 100,
    medium: (stats.priorityBreakdown.medium / stats.totalEvents) * 100,
    low: (stats.priorityBreakdown.low / stats.totalEvents) * 100
  };
  
  // Calculate speed statistics
  if (stats.speeds.length > 0) {
    stats.avgSpeed = stats.speeds.reduce((a, b) => a + b, 0) / stats.speeds.length;
    stats.maxSpeed = Math.max(...stats.speeds);
    stats.minSpeed = Math.min(...stats.speeds);
  } else {
    stats.avgSpeed = 0;
    stats.maxSpeed = 0;
    stats.minSpeed = 0;
  }
  
  return stats;
}

function getEmptyStats() {
  return {
    totalEvents: 0,
    uniqueTracks: 0,
    motionStateBreakdown: { STATIONARY: 0, WALKING: 0, MOVING: 0, LOITERING: 0 },
    motionStatePercentages: { STATIONARY: 0, WALKING: 0, MOVING: 0, LOITERING: 0 },
    priorityBreakdown: { high: 0, medium: 0, low: 0 },
    priorityPercentages: { high: 0, medium: 0, low: 0 },
    zoneActivity: {},
    avgSpeed: 0,
    maxSpeed: 0,
    minSpeed: 0,
    minTime: 0,
    maxTime: 0
  };
}

function displayZoneHotspots(zoneActivity) {
  const container = document.getElementById('zoneHotspots');
  if (!container) return;
  
  const zones = Object.entries(zoneActivity).sort((a, b) => b[1] - a[1]);
  
  if (zones.length === 0) {
    container.innerHTML = '<div style=\"text-align:center;padding:8px;background:white;border-radius:4px;font-size:11px;color:#718096;\">No zone data</div>';
    return;
  }
  
  const maxActivity = zones[0][1];
  
  container.innerHTML = zones.map(([zone, count]) => {
    const intensity = count / maxActivity;
    const bgColor = intensity > 0.7 ? '#e53e3e' : 
                    intensity > 0.4 ? '#f6ad55' : 
                    intensity > 0.2 ? '#48bb78' : '#667eea';
    
    return `
      <div style=\"background:${bgColor};color:white;padding:10px;border-radius:6px;text-align:center;\">
        <div style=\"font-size:16px;font-weight:bold;\">${zone}</div>
        <div style=\"font-size:12px;opacity:0.9;\">${count} events</div>
        <div style=\"font-size:10px;opacity:0.8;\">${(intensity * 100).toFixed(0)}%</div>
      </div>
    `;
  }).join('');
}

function displaySalientFrames(frames){
  console.log('[displaySalientFrames] Called with', frames ? frames.length : 0, 'frames');
  
  if(!Array.isArray(frames)) {
    console.warn('[displaySalientFrames] frames is not an array:', frames);
    return;
  }
  
  frames.slice(0,3).forEach((f,idx)=>{
    console.log(`[displaySalientFrames] Processing frame ${idx}:`, f);
    
    const img=document.getElementById(`frame${idx}`);
    const info=document.getElementById(`frame${idx}-info`);
    const objs=document.getElementById(`frame${idx}-objects`);
    
    const url=f.image_url||f.image_path||'';
    console.log(`[displaySalientFrames] Frame ${idx} URL:`, url);
    
    if(img){
      img.src=url; 
      img.style.display=url?'block':'none';
      console.log(`[displaySalientFrames] Set img src for frame${idx}:`, url);
    } else {
      console.warn(`[displaySalientFrames] Image element frame${idx} not found`);
    }
    
    if(info){
      info.innerHTML=`Frame ${f.frame_id||idx} • ${(f.saliency*100||0).toFixed(0)}% • 👥 ${f.persons||0} • 📦 ${f.objects||0}`;
      console.log(`[displaySalientFrames] Set info for frame${idx}`);
    }
    
    const all = Array.isArray(f.all_objects)?f.all_objects:[];
    if(objs){
      const y=all.filter(o=>o.type==='yolo_detection').slice(0,5);
      const tr=all.filter(o=>o.type==='tracked_object').slice(0,5);
      const detections = y.length ? y.map(o=>`• ${o.class||'object'} (${(o.confidence*100||0).toFixed(0)}%)`).join('<br>') : 'None';
      const tracked = tr.length ? tr.map(o=>`• ${o.class||'object'} #${o.track_id||'?'}`).join('<br>') : 'None';
      objs.innerHTML=`<strong>Detected Objects:</strong><br>${detections}<br><br><strong>Tracked:</strong><br>${tracked}`;
      console.log(`[displaySalientFrames] Set objects for frame${idx}:`, y.length, 'detections,', tr.length, 'tracked');
    }
  });
  
  console.log('[displaySalientFrames] Completed displaying', frames.slice(0,3).length, 'frames');
}

function displayVLMAnalysis(vlm, insights){
  if(!vlm||!vlm.length) return;
  const r=vlm[0];
  
  // Display scene description
  if(r.scene){
    const scene = r.scene;
    document.getElementById('surveillanceText').innerHTML = 
      `<strong>Environment:</strong> ${scene.type||'unknown'}<br>` +
      `<strong>Lighting:</strong> ${scene.lighting||'n/a'}<br>` +
      `<strong>Time:</strong> ${scene.time_of_day||'n/a'}`;
  }
  
  // Display risks and anomalies from insights
  if(insights){
    const risks = insights.risks || [];
    const anomalies = insights.anomalies || [];
    const overallRisk = insights.overall_risk || 'low';
    const detectedObjects = insights.detected_objects || [];
    
    // Risk display with ratings
    const riskColor = overallRisk === 'high' ? '#e53e3e' : 
                     overallRisk === 'medium' ? '#ed8936' : '#38a169';
    
    let riskHtml = `
      <div style="padding:10px; background:${riskColor}; color:white; border-radius:5px; margin-bottom:10px;">
        <strong>Overall Risk Level:</strong> ${overallRisk.toUpperCase()}
    `;
    
    // Add overall risk rating if available from VLM
    if(r.overall_risk_rating){
      riskHtml += ` (${r.overall_risk_rating}/10)`;
    }
    
    riskHtml += `</div><strong>Identified Risks (${risks.length}):</strong><br>`;
    
    if(risks.length){
      riskHtml += risks.map(risk=>{
        const rating = risk.rating ? ` [${risk.rating}/10]` : '';
        return `• <span style="color:${risk.severity==='high'?'#e53e3e':risk.severity==='medium'?'#ed8936':'#38a169'}">[${risk.severity||'low'}${rating}]</span> ${risk.type||'unknown'}: ${risk.description||''}`;
      }).join('<br>');
    } else {
      riskHtml += 'No risks detected';
    }
    
    // Anomaly display with ratings
    let anomalyHtml = '<br><br><strong>Detected Anomalies (${anomalies.length}):</strong>';
    
    if(r.overall_anomaly_rating){
      anomalyHtml += ` (Overall: ${r.overall_anomaly_rating}/10)`;
    }
    
    anomalyHtml += '<br>';
    
    if(anomalies.length){
      anomalyHtml += anomalies.map(a=>{
        const rating = a.rating ? ` [${a.rating}/10]` : '';
        return `• ${a.type||'unknown'}${rating}: ${a.description||''}`;
      }).join('<br>');
    } else {
      anomalyHtml += 'No anomalies detected';
    }
    
    // Objects display
    const objectsHtml = `
      <strong>Detected Object Types:</strong><br>
      ${detectedObjects.length ? detectedObjects.map(o=>`<span style="background:#667eea; color:white; padding:2px 8px; border-radius:3px; margin:2px; display:inline-block;">${o}</span>`).join(' ') : 'None'}
    `;
    
    document.getElementById('behavioralText').innerHTML = riskHtml + anomalyHtml;
    
    // Add objects to KG panel
    if(r.knowledge_graph){
      const k=r.knowledge_graph;
      const nodes=Array.isArray(k.nodes)?k.nodes:[], rels=Array.isArray(k.relationships)?k.relationships:[];
      document.getElementById('kgContent').innerHTML =
        objectsHtml + '<br><br>' +
        `<strong>Graph Nodes (${nodes.length}):</strong><br>${nodes.slice(0,8).map(n=>`• ${n.id} [${n.type}]`).join('<br>')}<br><br><strong>Relationships (${rels.length}):</strong><br>${rels.slice(0,8).map(x=>`${x.source} → ${x.target}`).join('<br>')}`;
    } else {
      document.getElementById('kgContent').innerHTML = objectsHtml;
    }
  }
  
  // Fallback to old structure
  if(r.surveillance_description && !r.scene) {
    document.getElementById('surveillanceText').innerHTML = r.surveillance_description;
  }
  if(r.behavioral_assessment && !insights){
    const b=r.behavioral_assessment;
    document.getElementById('behavioralText').innerHTML =
      `<strong>Risk:</strong> ${b.risk_level||'n/a'}<br><strong>Intent:</strong> ${b.inferred_intent||'n/a'}<br><strong>Subjects:</strong> ${(b.primary_subjects||[]).join(', ')||'n/a'}`;
  }
}

function switchIntelTab(tab){
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`${tab}-panel`).classList.add('active');
}

function setQuery(q){document.getElementById('ragQuery').value=q;}

async function performRAGSearch(){
  const q=document.getElementById('ragQuery').value.trim();
  const out=document.getElementById('ragResults');
  if(!q){out.textContent='Enter a question.';return;}
  out.textContent='Searching...';
  try{
    const resp=await fetch(`/rag/search?q=${encodeURIComponent(q)}&k=5&mode=llm`);
    const data=await resp.json();
    if(!resp.ok) throw new Error(data.error||'Search failed');
    out.innerHTML = `<div><strong>Answer:</strong> ${data.answer||'No direct answer'}</div>
                     <div><strong>Confidence:</strong> ${(data.confidence*100||0).toFixed(0)}%</div>
                     <div><strong>Evidence:</strong><br>${(data.evidence||[]).map(e=>`• ${e.file||''}: ${e.snippet||''}`).join('<br>')}</div>`;
    addToHistory(q);
  }catch(e){out.textContent=e.message;}
}

function addToHistory(q){
  queryHistory.unshift(q); if(queryHistory.length>6) queryHistory.pop();
  const h=document.getElementById('queryHistory');
  h.innerHTML=queryHistory.map(x=>`<div class="history-item" onclick="setQuery('${x.replace(/'/g,"\\'")}')">${x}</div>`).join('')||'No queries yet';
}

async function askQuestion() {
    const query = document.getElementById('rag-query').value;
    const ragType = document.querySelector('input[name="rag-type"]:checked').value; // Add radio buttons
    
    if (!query) return;
    
    const endpoint = ragType === 'graph' ? '/rag/graph' : 
                     ragType === 'hybrid' ? '/rag/hybrid' : '/rag/search';
    
    const response = await fetch(`${endpoint}?q=${encodeURIComponent(query)}`);
    const data = await response.json();
    
    // Display chain-of-thought if available
    if (data.chain_of_thought) {
        displayChainOfThought(data.chain_of_thought);
    }
    
    displayAnswer(data.answer, data.confidence, data.evidence);
}

function displayChainOfThought(steps) {
    let html = '<div class="cot-container"><h4>🧠 Chain of Thought Reasoning:</h4>';
    steps.forEach(step => {
        html += `
        <div class="cot-step">
            <strong>Step ${step.step}:</strong> ${step.reasoning}
            <ul>${step.findings.map(f => `<li>${f}</li>`).join('')}</ul>
        </div>`;
    });
    html += '</div>';
    document.getElementById('cot-display').innerHTML = html;
}