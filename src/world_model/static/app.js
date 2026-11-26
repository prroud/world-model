const socketUrl = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/dream`;
const connectionDot = document.querySelector('#connection-dot');
const connectionLabel = document.querySelector('#connection-label');
const frameElement = document.querySelector('#dream-frame');
const referenceElement = document.querySelector('#reference-frame');
const referenceView = document.querySelector('#reference-view');
const screenWrap = document.querySelector('#screen-wrap');
const placeholder = document.querySelector('#screen-placeholder');
const resetButton = document.querySelector('#reset-button');
const compareButton = document.querySelector('#compare-button');
const recordButton = document.querySelector('#record-button');
const temperatureSlider = document.querySelector('#temperature');
const temperatureValue = document.querySelector('#temperature-value');
const fpsValue = document.querySelector('#fps-value');
const frameCounter = document.querySelector('#frame-counter');
const modelState = document.querySelector('#model-state');
const inputState = document.querySelector('#input-state');
const latentCanvas = document.querySelector('#latent-chart');
const latentContext = latentCanvas.getContext('2d');
const latentSliders = document.querySelector('#latent-sliders');
const recordCanvas = document.querySelector('#record-canvas');
const recordContext = recordCanvas.getContext('2d');
const seedEpisodeSelect = document.querySelector('#seed-episode');
const seedFrameSlider = document.querySelector('#seed-frame');
const seedFrameValue = document.querySelector('#seed-frame-value');

const heldKeys = new Set();
let socket;
let hasReset = false;
let actionPending = false;
let frameCount = 0;
let previousFrameTime = 0;
let fpsSamples = [];
let currentLatent = [];
let comparisonEnabled = false;
let lastLatentEditAt = 0;
let latentEditTimer;
let seedEpisodes = [];
let mediaRecorder;
let recordingStream;
let recordedChunks = [];
let recordingSession;
let recordingStartedAt = 0;
const recordingSupported = typeof MediaRecorder !== 'undefined'
  && typeof recordCanvas.captureStream === 'function';

if (!recordingSupported) {
  recordButton.title = 'This browser does not support WebM recording.';
}

function setConnection(state, label) {
  connectionDot.classList.toggle('connected', state === 'connected');
  connectionDot.classList.toggle('error', state === 'error');
  connectionLabel.textContent = label;
  resetButton.disabled = state !== 'connected';
  compareButton.disabled = state !== 'connected';
  recordButton.disabled = state !== 'connected' || !hasReset || !recordingSupported;
}

function connect() {
  setConnection('connecting', 'Connecting');
  socket = new WebSocket(socketUrl);
  socket.addEventListener('open', () => {
    setConnection('connected', 'Connected');
    modelState.textContent = 'Ready';
    send({ type: 'seed_options' });
  });
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'seed_options') {
      seedEpisodes = message.episodes || [];
      seedEpisodeSelect.replaceChildren();
      seedEpisodes.forEach((episode) => {
        const option = document.createElement('option');
        option.value = episode.episode_index;
        option.textContent = `Episode ${episode.episode_index + 1} · ${episode.frame_count} frames`;
        seedEpisodeSelect.append(option);
      });
      seedEpisodeSelect.disabled = seedEpisodes.length === 0;
      seedFrameSlider.disabled = seedEpisodes.length === 0;
      if (seedEpisodes.length) {
        seedEpisodeSelect.value = String(seedEpisodes[0].episode_index);
        updateSeedFrameRange(Math.floor(seedEpisodes[0].frame_count / 2));
        resetDream();
      } else {
        modelState.textContent = 'No seed episodes';
      }
    } else if (message.type === 'frame') {
      frameElement.src = `data:image/png;base64,${message.image}`;
      frameElement.classList.add('visible');
      placeholder.classList.add('hidden');
      hasReset = true;
      recordButton.disabled = !recordingSupported || socket.readyState !== WebSocket.OPEN;
      actionPending = false;
      frameCount += 1;
      frameCounter.textContent = `${frameCount} FRAMES`;
      if (Array.isArray(message.latent)) updateLatent(message.latent);
      if (message.reference_image) showReferenceFrame(message.reference_image);
      const now = performance.now();
      if (previousFrameTime > 0) {
        fpsSamples.push(1000 / (now - previousFrameTime));
        if (fpsSamples.length > 20) fpsSamples.shift();
        fpsValue.textContent = (fpsSamples.reduce((sum, value) => sum + value, 0) / fpsSamples.length).toFixed(1);
      }
      previousFrameTime = now;
      modelState.textContent = 'Dreaming';
    } else if (message.type === 'temperature') {
      temperatureValue.textContent = Number(message.value).toFixed(2);
    } else if (message.type === 'latent') {
      updateLatent(message.latent);
      frameElement.src = `data:image/png;base64,${message.image}`;
    } else if (message.type === 'comparison') {
      comparisonEnabled = message.enabled;
      compareButton.disabled = false;
      compareButton.classList.toggle('active', comparisonEnabled);
      compareButton.innerHTML = comparisonEnabled
        ? '<span aria-hidden="true">⇄</span> Hide comparison'
        : '<span aria-hidden="true">⇄</span> Compare Gymnasium';
      screenWrap.classList.toggle('compare-mode', comparisonEnabled);
      referenceView.hidden = !comparisonEnabled;
      if (message.reference_image) showReferenceFrame(message.reference_image);
    } else if (message.type === 'error') {
      actionPending = false;
      compareButton.disabled = socket?.readyState !== WebSocket.OPEN;
      modelState.textContent = 'Error';
      placeholder.querySelector('span').textContent = message.message;
      placeholder.classList.remove('hidden');
      frameElement.classList.remove('visible');
    }
  });
  socket.addEventListener('close', () => {
    setConnection('error', 'Disconnected');
    hasReset = false;
    recordButton.disabled = true;
    actionPending = false;
    comparisonEnabled = false;
    screenWrap.classList.remove('compare-mode');
    referenceView.hidden = true;
    modelState.textContent = 'Offline';
    if (mediaRecorder?.state === 'recording') stopRecording();
  });
  socket.addEventListener('error', () => setConnection('error', 'Connection error'));
}

function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

function resetDream() {
  if (socket?.readyState !== WebSocket.OPEN) return;
  if (!seedEpisodes.length) return;
  hasReset = false;
  recordButton.disabled = true;
  actionPending = true;
  modelState.textContent = 'Resetting';
  placeholder.querySelector('span').textContent = 'Loading a real seed frame';
  placeholder.classList.remove('hidden');
  if (mediaRecorder?.state === 'recording') {
    recordSessionEvent('reset', {
      episode_index: Number(seedEpisodeSelect.value),
      frame_index: Number(seedFrameSlider.value),
    });
  }
  send({
    type: 'reset',
    episode_index: Number(seedEpisodeSelect.value),
    frame_index: Number(seedFrameSlider.value),
  });
}

function updateSeedFrameRange(preferredFrame = 0) {
  const selectedIndex = Number(seedEpisodeSelect.value);
  const episode = seedEpisodes.find((item) => item.episode_index === selectedIndex);
  if (!episode) return;
  seedFrameSlider.max = Math.max(0, episode.frame_count - 1);
  seedFrameSlider.value = Math.min(preferredFrame, Number(seedFrameSlider.max));
  seedFrameValue.value = `${Number(seedFrameSlider.value) + 1} / ${episode.frame_count}`;
}

function showReferenceFrame(image) {
  referenceElement.src = `data:image/png;base64,${image}`;
  referenceElement.classList.add('visible');
}

function drawRecordingFrame() {
  if (!frameElement.complete || !frameElement.naturalWidth) return;
  const { width, height } = recordCanvas;
  recordContext.fillStyle = '#101317';
  recordContext.fillRect(0, 0, width, height);
  recordContext.fillStyle = '#171b20';
  recordContext.fillRect(0, 0, width, 34);
  recordContext.fillRect(0, height - 38, width, 38);

  const frameSize = Math.min(height - 84, width - 48);
  const x = Math.round((width - frameSize) / 2);
  const y = Math.round((height - frameSize) / 2);
  recordContext.imageSmoothingEnabled = false;
  recordContext.drawImage(frameElement, x, y, frameSize, frameSize);

  recordContext.fillStyle = '#c8f169';
  recordContext.font = '500 12px monospace';
  recordContext.textBaseline = 'middle';
  recordContext.fillText('WORLD MODEL  /  DREAM DRIVE', 16, 17);
  recordContext.fillStyle = '#b5bec7';
  recordContext.font = '11px monospace';
  recordContext.fillText(
    `INPUT ${inputState.textContent.toUpperCase()}   TEMP ${Number(temperatureSlider.value).toFixed(2)}`,
    16,
    height - 19,
  );
  recordContext.textAlign = 'right';
  recordContext.fillStyle = '#89929c';
  recordContext.fillText(`FPS ${fpsValue.textContent}`, width - 16, height - 19);
  recordContext.textAlign = 'left';
}

frameElement.addEventListener('load', drawRecordingFrame);

function startRecording() {
  if (!recordingSupported || !hasReset) return;
  const mimeTypes = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = typeof MediaRecorder.isTypeSupported === 'function'
    ? mimeTypes.find((type) => MediaRecorder.isTypeSupported(type))
    : undefined;
  try {
    recordingStream = recordCanvas.captureStream(20);
    const recorder = mimeType
      ? new MediaRecorder(recordingStream, { mimeType })
      : new MediaRecorder(recordingStream);
    mediaRecorder = recorder;
    recordedChunks = [];
    const startedAt = new Date();
    recordingStartedAt = performance.now();
    recordingSession = {
      schema_version: 1,
      started_at: startedAt.toISOString(),
      browser: navigator.userAgent,
      video: { format: 'webm', capture_fps: 20, view: 'world_model_only', audio: false },
      action_format: ['steering', 'gas', 'brake'],
      initial_temperature: Number(temperatureSlider.value),
      start_seed: {
        episode_index: Number(seedEpisodeSelect.value),
        frame_index: Number(seedFrameSlider.value),
      },
      events: [],
    };
    recordSessionEvent('recording_started');
    recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    });
    recorder.addEventListener('stop', () => {
      const blob = new Blob(recordedChunks, { type: recorder.mimeType || 'video/webm' });
      if (blob.size > 0) {
        const filename = `world-model-dream-${startedAt.toISOString().replace(/[:.]/g, '-')}`;
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${filename}.webm`;
        document.body.append(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
        downloadJson(`${filename}.json`, recordingSession);
        modelState.textContent = 'Recording saved';
      } else {
        modelState.textContent = 'Recording empty';
      }
      recordingStream?.getTracks().forEach((track) => track.stop());
      recordingStream = undefined;
      recordedChunks = [];
      recordingSession = undefined;
      recordButton.classList.remove('recording');
      recordButton.setAttribute('aria-pressed', 'false');
      recordButton.innerHTML = '<span class="record-indicator" aria-hidden="true"></span> Record session';
      recordButton.disabled = !recordingSupported || !hasReset || socket?.readyState !== WebSocket.OPEN;
    }, { once: true });
    recorder.start(1000);
    recordButton.classList.add('recording');
    recordButton.setAttribute('aria-pressed', 'true');
    recordButton.innerHTML = '<span class="record-indicator" aria-hidden="true"></span> Stop recording';
    drawRecordingFrame();
  } catch (error) {
    modelState.textContent = 'Recording unavailable';
    console.error('Could not start browser recording:', error);
    recordingStream?.getTracks().forEach((track) => track.stop());
    recordingStream = undefined;
    recordingSession = undefined;
  }
}

function stopRecording() {
  if (mediaRecorder?.state === 'recording') {
    recordButton.disabled = true;
    modelState.textContent = 'Saving recording';
    recordSessionEvent('recording_stopped');
    mediaRecorder.stop();
  }
}

function recordSessionEvent(type, values = {}) {
  if (!recordingSession) return;
  recordingSession.events.push({
    time_ms: Math.round(performance.now() - recordingStartedAt),
    type,
    ...values,
  });
}

function downloadJson(filename, value) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function toggleRecording() {
  if (mediaRecorder?.state === 'recording') stopRecording();
  else startRecording();
}

function updateLatent(values) {
  if (!Array.isArray(values)) return;
  if (Date.now() - lastLatentEditAt > 300) currentLatent = values;
  drawLatent(currentLatent);
  latentSliders.querySelectorAll('input').forEach((slider, index) => {
    if (document.activeElement === slider) return;
    slider.value = currentLatent[index] ?? 0;
    slider.previousElementSibling.querySelector('output').value = Number(currentLatent[index] ?? 0).toFixed(2);
  });
}

function createLatentSliders() {
  for (let index = 0; index < 32; index += 1) {
    const control = document.createElement('div');
    control.className = 'latent-control';
    const label = document.createElement('label');
    label.htmlFor = `latent-${index}`;
    label.textContent = `z${index}`;
    const output = document.createElement('output');
    output.value = '0.00';
    const slider = document.createElement('input');
    slider.id = `latent-${index}`;
    slider.type = 'range';
    slider.min = '-3';
    slider.max = '3';
    slider.step = '0.01';
    slider.value = '0';
    slider.addEventListener('input', () => {
      currentLatent[index] = Number(slider.value);
      output.value = Number(slider.value).toFixed(2);
      lastLatentEditAt = Date.now();
      drawLatent(currentLatent);
      clearTimeout(latentEditTimer);
      latentEditTimer = setTimeout(() => {
        recordSessionEvent('latent_edit', { values: [...currentLatent] });
        send({ type: 'latent', values: currentLatent });
      }, 80);
    });
    label.append(output);
    control.append(label, slider);
    latentSliders.append(control);
  }
}

function toggleComparison() {
  if (socket?.readyState !== WebSocket.OPEN) return;
  compareButton.disabled = true;
  recordSessionEvent('comparison', { enabled: !comparisonEnabled });
  send({ type: 'compare', enabled: !comparisonEnabled });
}

function currentAction() {
  const left = heldKeys.has('ArrowLeft') || heldKeys.has('a');
  const right = heldKeys.has('ArrowRight') || heldKeys.has('d');
  const gas = heldKeys.has('ArrowUp') || heldKeys.has('w');
  const brake = heldKeys.has('ArrowDown') || heldKeys.has('s');
  return [Number(right) - Number(left), Number(gas), Number(brake)];
}

function updateInputLabel(action) {
  const [steering, gas, brake] = action;
  const labels = [];
  if (steering < 0) labels.push('Left');
  if (steering > 0) labels.push('Right');
  if (gas) labels.push('Gas');
  if (brake) labels.push('Brake');
  inputState.textContent = labels.length ? labels.join(' + ') : 'Idle';
}

function drawLatent(values) {
  const { width, height } = latentCanvas;
  latentContext.clearRect(0, 0, width, height);
  latentContext.strokeStyle = '#353c43';
  latentContext.beginPath();
  latentContext.moveTo(0, height / 2);
  latentContext.lineTo(width, height / 2);
  latentContext.stroke();
  if (!values.length) return;
  const barWidth = width / values.length;
  values.forEach((value, index) => {
    const normalized = Math.max(-3, Math.min(3, value)) / 3;
    const barHeight = Math.abs(normalized) * (height / 2 - 5);
    latentContext.fillStyle = normalized >= 0 ? '#c8f169' : '#78a7c9';
    latentContext.fillRect(
      index * barWidth + 2,
      normalized >= 0 ? height / 2 - barHeight : height / 2,
      Math.max(2, barWidth - 4),
      barHeight,
    );
  });
}

window.addEventListener('keydown', (event) => {
  if (event.target instanceof HTMLInputElement && event.target.type === 'range') return;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'a', 'd', 'w', 's'].includes(key)) {
    event.preventDefault();
    heldKeys.add(key);
  }
});
window.addEventListener('keyup', (event) => {
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  heldKeys.delete(key);
});
window.addEventListener('blur', () => heldKeys.clear());

temperatureSlider.addEventListener('input', () => {
  temperatureValue.textContent = Number(temperatureSlider.value).toFixed(2);
  recordSessionEvent('temperature', { value: Number(temperatureSlider.value) });
  send({ type: 'temperature', value: Number(temperatureSlider.value) });
});
resetButton.addEventListener('click', resetDream);
seedEpisodeSelect.addEventListener('change', () => {
  const selected = seedEpisodes.find((item) => item.episode_index === Number(seedEpisodeSelect.value));
  updateSeedFrameRange(selected ? Math.floor(selected.frame_count / 2) : 0);
  resetDream();
});
seedFrameSlider.addEventListener('input', () => {
  const episode = seedEpisodes.find((item) => item.episode_index === Number(seedEpisodeSelect.value));
  if (episode) seedFrameValue.value = `${Number(seedFrameSlider.value) + 1} / ${episode.frame_count}`;
});
seedFrameSlider.addEventListener('change', resetDream);
compareButton.addEventListener('click', toggleComparison);
recordButton.addEventListener('click', toggleRecording);

setInterval(() => {
  if (!hasReset || actionPending || socket?.readyState !== WebSocket.OPEN) return;
  const action = currentAction();
  updateInputLabel(action);
  actionPending = true;
  recordSessionEvent('action', { action });
  send({ type: 'action', action, temperature: Number(temperatureSlider.value) });
}, 50);

drawLatent(currentLatent);
createLatentSliders();
connect();
