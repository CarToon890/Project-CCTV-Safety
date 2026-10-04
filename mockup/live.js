/* Real single-stream pilot client. Mock cameras elsewhere remain labelled as mock data. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const sourceType = $('liveSourceType'), source = $('liveSource'), device = $('liveDevice');
  const start = $('liveStart'), stop = $('liveStop'), status = $('liveRealStatus');
  const preview = $('liveRealPreview'), events = $('liveRealEvents'), sourceHelp = $('liveSourceHelp');
  if (!start) return;
  let socket = null;
  function report(text) { status.textContent = text; }
  function addEvent(text) {
    const row = document.createElement('div'); row.className = 'feed-item'; row.textContent = text;
    events.prepend(row); while (events.children.length > 40) events.lastChild.remove();
  }
  async function request(path, method) {
    const r = await fetch(path, { method: method || 'GET', headers: {'Content-Type': 'application/json'} });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error?.message || 'Live API error');
    return body;
  }
  function updateSourceHelp() {
    const isFile = sourceType.value === 'file';
    source.placeholder = isFile ? 'data/raw/clip.mp4' : 'rtsp://กล้องในเครือข่ายของคุณ/stream';
    if (sourceHelp) sourceHelp.textContent = isFile
      ? 'ระบุ path ของคลิปภายในโฟลเดอร์ data/ เช่น data/raw/clip.mp4 · คลิปจะถูก replay เพื่อทดสอบ ไม่ใช่กล้องสด'
      : 'ใส่ RTSP URL ของกล้องที่เครื่องนี้เข้าถึงได้ · อย่าแชร์ URL เพราะอาจมีข้อมูลเข้าสู่ระบบ';
  }
  sourceType.addEventListener('change', updateSourceHelp);
  updateSourceHelp();
  start.addEventListener('click', async () => {
    if (!source.value.trim()) {
      report(sourceType.value === 'file' ? 'กรุณาระบุ path ของคลิปในโฟลเดอร์ data/' : 'กรุณาระบุ RTSP URL');
      source.focus();
      return;
    }
    start.disabled = true; events.replaceChildren(); preview.hidden = true;
    try {
      const result = await fetch('/api/live/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
        source_type:sourceType.value, source:source.value, model:$('liveModel').value, device:device.value
      })});
      const body = await result.json();
      if (!result.ok) throw new Error(body.error?.message || 'เริ่ม Live ไม่สำเร็จ');
      stop.disabled = false; report('กำลังทำงาน · device: ' + body.device_used);
      socket = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/api/live/events');
      socket.onmessage = ev => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'preview' && typeof msg.jpeg_b64 === 'string') {
          preview.src = 'data:image/jpeg;base64,' + msg.jpeg_b64; preview.hidden = false;
        } else if (msg.type === 'candidate') addEvent('Candidate เบื้องต้น · ' + msg.time_s + 's · latency ' + msg.processing_latency_ms + 'ms · ' + msg.trigger_reasons.join(', '));
        else if (msg.type === 'x3d') addEvent('X3D ' + msg.label + ' · fight=' + msg.fight + ' · ' + msg.start_s + '–' + msg.end_s + 's');
        else if (msg.type === 'health') {
          report(msg.state + (msg.reason ? ' · ' + msg.reason : '') + ' · scanned ' + msg.frames_face_scanned + '/' + msg.frames_received);
          if (['stopped','degraded','error'].includes(msg.state)) { start.disabled = false; stop.disabled = true; }
        }
      };
      socket.onerror = () => report('WebSocket ขาดการเชื่อมต่อ · ตรวจ /api/live/status');
    } catch (err) { report(err.message); start.disabled = false; }
  });
  stop.addEventListener('click', async () => {
    stop.disabled = true;
    try { await request('/api/live/stop', 'POST'); report('กำลังหยุดและระบายเฟรมที่รับแล้ว'); }
    catch (err) { report(err.message); stop.disabled = false; }
  });
}());
