const $ = id => document.getElementById(id);
const history = {fps: [], temp: []};
const value = (v, fallback = '—') => v === null || v === undefined ? fallback : v;
const esc = v => String(v).replace(/[&<>"']/g, c => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[c]));

const bytes = n => {
  if (n === null || n === undefined) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB'];
  let index = 0;
  while (n >= 1024 && index < units.length - 1) {
    n /= 1024;
    index += 1;
  }
  return `${n.toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
};

function setText(id, text) {
  $(id).textContent = text;
}

function drawChart() {
  const canvas = $('performance-chart');
  const context = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = '#dce7e1';
  context.lineWidth = 1;
  for (let index = 1; index < 5; index += 1) {
    context.beginPath();
    context.moveTo(35, index * height / 5);
    context.lineTo(width - 10, index * height / 5);
    context.stroke();
  }

  function line(values, color, min, max) {
    if (values.length < 2) return;
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.beginPath();
    values.forEach((sample, index) => {
      const x = 35 + index * (width - 50) / 59;
      const y = height - 15 - (Math.max(min, Math.min(max, sample)) - min)
        * (height - 30) / (max - min);
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.stroke();
  }

  line(history.fps, '#22985a', 0, 20);
  line(history.temp, '#e6a100', 35, 90);
}

function updateFaceOverlay(driver) {
  const detected = driver.face_detected === true;
  const status = $('face-status');
  const box = $('face-box');
  status.textContent = detected ? 'FACE DETECTED' : 'NO FACE';
  status.className = `face-status ${detected ? 'face-found' : 'face-missing'}`;

  const bbox = driver.face_bbox_normalized;
  const valid = detected && Array.isArray(bbox) && bbox.length === 4
    && bbox.every(Number.isFinite) && bbox[2] > bbox[0] && bbox[3] > bbox[1];
  box.classList.toggle('visible', valid);
  box.setAttribute('aria-hidden', String(!valid));
  if (!valid) return;

  box.style.left = `${bbox[0] * 100}%`;
  box.style.top = `${bbox[1] * 100}%`;
  box.style.width = `${(bbox[2] - bbox[0]) * 100}%`;
  box.style.height = `${(bbox[3] - bbox[1]) * 100}%`;
}

const behaviorDefinitions = [
  {source: 'phone', violation: 'PHONE_USE', chip: 'behavior-phone', state: 'behavior-phone-state', title: 'PHONE'},
  {source: 'cigarette', violation: 'SMOKING', chip: 'behavior-smoking', state: 'behavior-smoking-state', title: 'SMOKING'},
  {source: 'drink_or_food', violation: 'EATING', chip: 'behavior-eating', state: 'behavior-eating-state', title: 'EATING'},
];

function updateBehaviorDetection(behavior, driver) {
  const detections = Array.isArray(behavior.detections) ? behavior.detections : [];
  const confirmed = new Set([
    ...(behavior.active_behaviors || []),
    ...(driver.violations || []),
  ]);

  behaviorDefinitions.forEach(definition => {
    const detection = detections
      .filter(item => item.label === definition.source && Number.isFinite(item.confidence))
      .sort((left, right) => right.confidence - left.confidence)[0];
    const isConfirmed = confirmed.has(definition.violation);
    const chip = $(definition.chip);
    chip.className = `behavior-chip ${isConfirmed ? 'confirmed' : (detection ? 'detected' : '')}`;
    setText(
      definition.state,
      isConfirmed
        ? 'CONFIRMED'
        : (detection ? `DETECTED ${Math.round(detection.confidence * 100)}%` : 'CLEAR'),
    );
  });

  const overlays = $('behavior-overlays');
  overlays.replaceChildren();
  detections.forEach(detection => {
    const definition = behaviorDefinitions.find(item => item.source === detection.label);
    const bbox = detection.bbox_normalized;
    const valid = definition && Array.isArray(bbox) && bbox.length === 4
      && bbox.every(Number.isFinite) && bbox[2] > bbox[0] && bbox[3] > bbox[1]
      && Number.isFinite(detection.confidence);
    if (!valid) return;

    const box = document.createElement('div');
    box.className = `behavior-box behavior-box-${detection.label}`;
    box.style.left = `${bbox[0] * 100}%`;
    box.style.top = `${bbox[1] * 100}%`;
    box.style.width = `${(bbox[2] - bbox[0]) * 100}%`;
    box.style.height = `${(bbox[3] - bbox[1]) * 100}%`;
    const label = document.createElement('span');
    label.textContent = `${definition.title} ${Math.round(detection.confidence * 100)}%`;
    box.appendChild(label);
    overlays.appendChild(box);
  });
}

function updateStatus(snapshot) {
  const runtime = snapshot.runtime || {};
  const driver = snapshot.driver || {};
  const behavior = snapshot.behavior || {};
  const camera = snapshot.camera || {};
  const perf = snapshot.performance || {};
  const system = snapshot.system || {};
  const power = snapshot.power || {};
  const models = snapshot.models || {};
  const recording = snapshot.recording || {};
  const dashboard = snapshot.dashboard || {};
  const queues = perf.queues || {};
  const latency = perf.latency || {};
  const flags = system.throttled || {};

  const state = driver.state || 'STARTING';
  setText('driver-state', state);
  const card = $('state-card');
  card.className = 'state-card panel ' + (state === 'CRITICAL'
    ? 'state-critical'
    : (['DROWSY', 'FATIGUE_WARNING'].includes(state) ? 'state-warning' : ''));
  const violations = driver.violations || [];
  setText('active-violations', violations.length
    ? violations.join(' · ')
    : 'No active violations');
  const rawProbability = driver.probability;
  const probability = rawProbability === null || rawProbability === undefined
    ? NaN
    : Number(rawProbability);
  setText('model-probability', Number.isFinite(probability)
    ? `${Math.round(probability * 100)}%`
    : '—');
  $('probability-fill').style.width = Number.isFinite(probability)
    ? `${Math.max(0, Math.min(100, probability * 100))}%`
    : '0';

  updateFaceOverlay(driver);
  updateBehaviorDetection(behavior, driver);
  setText('blink-rate', String(Number(driver.blink_count_60s || 0).toFixed(0)));
  setText('camera-resolution', camera.width ? `${camera.width} × ${camera.height}` : '—');
  setText('camera-fps', `${Number(perf.capture_fps || 0).toFixed(1)} FPS`);
  setText('capture-drop', `${camera.dropped_frames || 0} dropped`);
  setText('ai-fps', Number(perf.effective_fps || 0).toFixed(1));
  setText('capture-fps', `${Number(perf.capture_fps || 0).toFixed(1)} fps`);
  setText('capture-queue', `queue ${queues.capture || 0}`);
  setText('core-fps', `${Number(perf.ai_fps || 0).toFixed(1)} fps`);
  setText('yolo-fps', `${Number(perf.yolo_fps || 0).toFixed(1)} fps`);
  setText('yolo-latency', `p95 ${Number((latency.yolo || {}).p95_ms || 0).toFixed(0)} ms · interval ${Number(perf.yolo_interval_sec || 0).toFixed(2)} s`);
  setText('recorder-fps', `${Number(perf.recorder_fps || 0).toFixed(1)} fps`);
  setText('recorder-drop', `${(recording.dropped_frames || 0) + (recording.hub_dropped_frames || 0)} dropped`);
  setText('preview-fps', `${Number(perf.preview_fps || 0).toFixed(1)} fps`);
  setText('preview-clients', `${dashboard.clients || 0} clients`);
  setText('decision-age', perf.decision_staleness_sec === null
    || perf.decision_staleness_sec === undefined
    ? '—'
    : `${Number(perf.decision_staleness_sec).toFixed(2)} s`);
  setText('fusion-latency', `Fusion p95 ${Number((latency.fusion || {}).p95_ms || 0).toFixed(2)} ms`);
  setText('temperature', system.temperature_c === null || system.temperature_c === undefined
    ? '—'
    : `${Number(system.temperature_c).toFixed(1)}°C`);
  setText('cpu-ram', `${Number(system.cpu_percent || 0).toFixed(0)}% / ${Number(system.ram_percent || 0).toFixed(0)}%`);
  const currentClock = system.cpu_frequency_mhz
    ? `${Number(system.cpu_frequency_mhz).toFixed(0)} MHz`
    : 'Clock unavailable';
  const maxClock = system.cpu_max_frequency_mhz
    ? ` / max ${Number(system.cpu_max_frequency_mhz).toFixed(0)}`
    : '';
  setText('cpu-clock', currentClock + maxClock);
  setText('storage-free', bytes((system.disk || {}).free_bytes));
  setText('incident-count', `${recording.incident_count || 0} incidents`);
  setText('power-mode', power.safe_mode ? `Safe mode · ${power.reason}` : 'Normal power mode');
  setText('safe-mode-label', power.safe_mode ? 'SAFE MODE' : 'NORMAL');
  setText('drowsiness-model', models.drowsiness || '—');
  setText('behavior-model', models.behavior || '—');
  setText('behavior-backend', models.behavior_backend
    ? `${models.behavior_backend} · ${models.behavior_runtime_version || 'runtime unknown'}`
    : '—');
  setText('feature-version', models.feature_version || '—');
  setText('fusion-version', models.fusion_version || '—');
  setText('drift-status', models.drift || '—');
  const activeFlags = Object.entries(flags)
    .filter(([key, flag]) => flag && key !== 'raw' && key !== 'hex')
    .map(([key]) => key.replaceAll('_', ' '));
  setText('power-flags', activeFlags.join(', ') || flags.hex || 'unavailable');
  setText('license-status', models.commercial_license_status || '—');
  const healthy = runtime.status === 'RUNNING' && !power.safe_mode;
  setText('health-label', power.safe_mode ? 'Power constrained' : (runtime.status || 'Starting'));
  $('health-dot').style.background = healthy
    ? '#22985a'
    : (runtime.status === 'FAILED' ? '#d43b45' : '#e6a100');

  history.fps.push(Number(perf.effective_fps || 0));
  history.temp.push(Number(system.temperature_c || 0));
  if (history.fps.length > 60) {
    history.fps.shift();
    history.temp.shift();
  }
  drawChart();
}

async function poll() {
  try {
    const response = await fetch('/api/v1/status', {cache: 'no-store'});
    if (!response.ok) throw Error(response.status);
    updateStatus(await response.json());
  } catch (error) {
    setText('health-label', 'Dashboard disconnected');
    $('health-dot').style.background = '#d43b45';
  }
}

async function incidents() {
  try {
    const response = await fetch('/api/v1/incidents?limit=20', {cache: 'no-store'});
    const data = await response.json();
    const body = $('incident-table');
    const rows = data.incidents || [];
    body.innerHTML = rows.length ? rows.map(incident => {
      const triggers = (incident.trigger_kinds || []).join(', ') || '—';
      const evidence = (incident.violations || []).join(', ') || '—';
      const duration = incident.duration_sec === null || incident.duration_sec === undefined
        ? '—'
        : esc(Number(incident.duration_sec).toFixed(1) + ' s');
      return `<tr><td>${esc(value(incident.started_utc, '—'))}</td><td>${esc(value(incident.highest_state, '—'))}</td><td>${esc(triggers)}</td><td>${esc(evidence)}</td><td>${duration}</td><td>${esc(bytes(incident.video_size_bytes))}</td></tr>`;
    }).join('') : '<tr><td colspan="6">No incidents saved</td></tr>';
  } catch (error) {
    // Keep the last good incident list while the runtime is busy or restarting.
  }
}

poll();
incidents();
setInterval(poll, 500);
setInterval(incidents, 10000);
