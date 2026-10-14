const $ = (id) => document.getElementById(id);
const ui = {
  audio: $('audio'), canvas: $('waveform'), waveWrap: $('waveWrap'),
  segmentsBody: $('segmentsBody'), rules: $('rules'), annotator: $('annotator'),
};
let session;
let currentIndex = 0;
let draft = null;
let peaks = null;
let selection = null;
let dragStart = null;
let saveTimer = null;
let metaTimer = null;
let saves = Promise.resolve();
let toastTimer = null;

function toast(message, error = false) {
  const node = $('toast');
  node.textContent = message;
  node.className = `visible${error ? ' error' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.className = '', 3600);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { message = (await response.json()).error || message; } catch (_) { /* keep status */ }
    throw new Error(message);
  }
  return response.json();
}

function clip() { return session.clips[currentIndex]; }
function readOnly() { return clip().phase === 'development' && session.state.rules_frozen; }
function roundTime(value) { return Math.round(Number(value) * 100) / 100; }
function formatTime(value) {
  const v = Math.max(0, Number(value) || 0);
  return `${String(Math.floor(v / 60)).padStart(2, '0')}:${(v % 60).toFixed(2).padStart(5, '0')}`;
}
function reviewedCount() { return session.clips.filter(c => session.state.clips[c.clip_id]?.reviewed).length; }
function currentRecord() {
  const saved = session.state.clips[clip().clip_id];
  return saved ? structuredClone(saved) : {
    segments: [], reviewed: false, difficult_to_judge: false,
    no_speech_confirmed: false, notes: '',
  };
}

async function openClip(index) {
  clearTimeout(saveTimer);
  if (metaTimer) await saveMeta();
  if (draft && !readOnly()) await persistClip(false);
  const target = session.clips[index];
  if (!target || (target.phase === 'audit' && !session.state.rules_frozen)) {
    toast('请先完成 D 阶段并固定规则', true);
    return;
  }
  ui.audio.pause();
  currentIndex = index;
  const loadedClipId = target.clip_id;
  draft = currentRecord();
  selection = null;
  peaks = null;
  ui.audio.src = `/api/audio/${encodeURIComponent(clip().clip_id)}`;
  ui.audio.load();
  $('seek').value = 0;
  $('playPause').textContent = '▶';
  $('timeReadout').textContent = `00:00.00 / ${formatTime(clip().duration_seconds)}`;
  render();
  try {
    const response = await fetch(ui.audio.src);
    if (!response.ok) throw new Error('音频无法载入');
    const data = await response.arrayBuffer();
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await context.decodeAudioData(data);
    if (clip().clip_id === loadedClipId) peaks = decoded.getChannelData(0);
    await context.close();
    if (clip().clip_id === loadedClipId) drawWave();
  } catch (error) {
    toast(`波形载入失败：${error.message}`, true);
  }
}

function render() {
  const c = clip();
  const isD = c.phase === 'development';
  const locked = readOnly();
  $('clipTitle').textContent = c.clip_id;
  $('phaseLabel').textContent = isD ? 'D · 制定规则' : 'A · 正式审计';
  $('progressText').textContent = `${currentIndex + 1} / ${session.clips.length}`;
  $('progressFill').style.width = `${reviewedCount() / session.clips.length * 100}%`;
  $('footerStatus').textContent = `已完成 ${reviewedCount()} / ${session.clips.length} 段`;
  $('previous').disabled = currentIndex === 0;
  $('next').disabled = currentIndex === session.clips.length - 1 || (!session.state.rules_frozen && session.clips[currentIndex + 1]?.phase === 'audit');
  $('stageGuide').innerHTML = isD
    ? '<strong>D 阶段：</strong>先独立听 8 段开发片段，形成你自己的边界判断规则。规则固定后才能进入 A 阶段。'
    : '<strong>A 阶段：</strong>按已固定的规则独立标注 16 段。不要查看自动边界、老师总分或预测结果。';
  ui.rules.value = session.state.rules;
  ui.rules.disabled = session.state.rules_frozen;
  ui.annotator.value = session.state.annotator;
  $('freeze').hidden = session.state.rules_frozen || reviewedCountD() !== 8;
  $('completeNext').disabled = locked;
  $('completeNext').innerHTML = currentIndex === session.clips.length - 1
    ? '完成最后一段 <span aria-hidden="true">✓</span>'
    : '完成本段并下一段 <span aria-hidden="true">→</span>';
  $('addSpeech').disabled = !selection || locked;
  $('addUncertain').disabled = !selection || locked;
  $('noSpeech').checked = Boolean(draft.no_speech_confirmed);
  $('difficult').checked = Boolean(draft.difficult_to_judge);
  $('clipNotes').value = draft.notes || '';
  $('noSpeech').disabled = locked;
  $('difficult').disabled = locked;
  $('clipNotes').disabled = locked;
  renderSegments();
  updateSelectionText();
  drawWave();
}

function reviewedCountD() {
  return session.clips.filter(c => c.phase === 'development' && session.state.clips[c.clip_id]?.reviewed).length;
}

function renderSegments() {
  $('segmentCount').textContent = `${draft.segments.length} 段`;
  if (!draft.segments.length) {
    ui.segmentsBody.innerHTML = '<tr><td colspan="6" class="empty">尚未标出语音。可以播放后在波形上拖选。</td></tr>';
    return;
  }
  ui.segmentsBody.replaceChildren();
  draft.segments.forEach((segment, index) => {
    const row = document.createElement('tr');
    const number = document.createElement('td');
    number.textContent = String(index + 1);
    row.append(number);
    for (const [field, label] of [['start_seconds', '开始时间'], ['end_seconds', '结束时间']]) {
      const cell = document.createElement('td');
      const input = document.createElement('input');
      input.type = 'number'; input.min = '0'; input.max = '30'; input.step = '0.01';
      input.value = Number(segment[field]).toFixed(2); input.disabled = readOnly();
      input.setAttribute('aria-label', `${index + 1} ${label}`);
      input.addEventListener('change', () => {
        segment[field] = roundTime(input.value);
        drawWave(); scheduleSave();
      });
      cell.append(input); row.append(cell);
    }
    const labelCell = document.createElement('td');
    const select = document.createElement('select');
    select.disabled = readOnly();
    [['speech', '语音'], ['uncertain', '不确定']].forEach(([value, text]) => {
      const option = document.createElement('option'); option.value = value; option.textContent = text; select.append(option);
    });
    select.value = segment.label;
    select.addEventListener('change', () => { segment.label = select.value; scheduleSave(); drawWave(); });
    labelCell.append(select); row.append(labelCell);
    const noteCell = document.createElement('td');
    const note = document.createElement('input');
    note.type = 'text'; note.maxLength = 500; note.placeholder = '可选备注'; note.value = segment.notes || '';
    note.disabled = readOnly();
    note.addEventListener('input', () => { segment.notes = note.value; scheduleSave(); });
    noteCell.append(note); row.append(noteCell);
    const action = document.createElement('td');
    const remove = document.createElement('button');
    remove.type = 'button'; remove.className = 'delete-button'; remove.textContent = '删除'; remove.disabled = readOnly();
    remove.addEventListener('click', () => { draft.segments.splice(index, 1); renderSegments(); drawWave(); scheduleSave(); });
    action.append(remove); row.append(action);
    ui.segmentsBody.append(row);
  });
}

function updateSelectionText() {
  $('selectionText').textContent = selection
    ? `已选 ${selection[0].toFixed(2)}–${selection[1].toFixed(2)} 秒（${(selection[1] - selection[0]).toFixed(2)} 秒）`
    : '在波形上拖动，选取一段可闻语音';
  $('addSpeech').disabled = !selection || readOnly();
  $('addUncertain').disabled = !selection || readOnly();
}

function waveTime(event) {
  const bounds = ui.canvas.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
  return roundTime(fraction * clip().duration_seconds);
}

function drawWave() {
  if (!session) return;
  const canvas = ui.canvas;
  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(1, Math.round(canvas.clientHeight));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#f7fbfd'; ctx.fillRect(0, 0, width, height);
  const top = 24, bottom = height - 12, middle = (top + bottom) / 2;
  for (let second = 0; second <= 30; second += 5) {
    const x = second / 30 * width;
    ctx.strokeStyle = '#dce8ef'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.fillStyle = '#6e8396'; ctx.font = '11px system-ui';
    ctx.fillText(String(second), Math.min(x + 3, width - 16), 15);
  }
  if (peaks) {
    const step = peaks.length / width;
    ctx.strokeStyle = '#439ba7'; ctx.lineWidth = 1;
    for (let x = 0; x < width; x++) {
      const from = Math.floor(x * step), to = Math.max(from + 1, Math.floor((x + 1) * step));
      let min = 0, max = 0;
      for (let i = from; i < to; i += Math.max(1, Math.floor(step / 8))) {
        const sample = peaks[i] || 0;
        min = Math.min(min, sample); max = Math.max(max, sample);
      }
      ctx.beginPath(); ctx.moveTo(x + .5, middle - max * (bottom - top) * .43);
      ctx.lineTo(x + .5, middle - min * (bottom - top) * .43); ctx.stroke();
    }
  } else {
    ctx.fillStyle = '#8196a7'; ctx.font = '13px system-ui';
    ctx.fillText('正在载入原声波形…', 20, middle);
  }
  for (const segment of draft.segments) {
    const x = segment.start_seconds / 30 * width;
    const w = (segment.end_seconds - segment.start_seconds) / 30 * width;
    ctx.fillStyle = segment.label === 'uncertain' ? 'rgba(217,159,50,.20)' : 'rgba(13,123,135,.17)';
    ctx.fillRect(x, top, w, bottom - top);
    ctx.strokeStyle = segment.label === 'uncertain' ? '#c58b2b' : '#0d7b87';
    ctx.strokeRect(x, top, w, bottom - top);
  }
  if (selection) {
    const x = selection[0] / 30 * width;
    const w = (selection[1] - selection[0]) / 30 * width;
    ctx.fillStyle = 'rgba(20,133,145,.20)'; ctx.fillRect(x, top, w, bottom - top);
    ctx.strokeStyle = '#0b7c87'; ctx.lineWidth = 2; ctx.strokeRect(x, top, w, bottom - top);
  }
  const playX = (ui.audio.currentTime || 0) / 30 * width;
  ctx.strokeStyle = '#213c54'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(playX, top); ctx.lineTo(playX, bottom); ctx.stroke();
}

function addSelection(label) {
  if (!selection || readOnly()) return;
  const [start_seconds, end_seconds] = selection;
  if (end_seconds - start_seconds < .02) { toast('选区太短，请重新拖动', true); return; }
  if (draft.segments.some(s => start_seconds < s.end_seconds && end_seconds > s.start_seconds)) {
    toast('选区与已有标注重叠，请调整范围', true); return;
  }
  draft.segments.push({ start_seconds, end_seconds, label, notes: '' });
  draft.segments.sort((a, b) => a.start_seconds - b.start_seconds);
  draft.no_speech_confirmed = false;
  $('noSpeech').checked = false;
  selection = null;
  renderSegments(); updateSelectionText(); drawWave(); scheduleSave();
}

function setSaveState(text) { $('saveState').textContent = text; }
function postClip(reviewed) {
  const c = clip();
  const snapshot = structuredClone({ ...draft, reviewed: reviewed || draft.reviewed });
  saves = saves.catch(() => {}).then(async () => {
    const result = await api(`/api/clip/${encodeURIComponent(c.clip_id)}`, {
      method: 'POST', body: JSON.stringify(snapshot),
    });
    session.state = result.state;
    setSaveState('已保存');
    $('progressFill').style.width = `${reviewedCount() / 24 * 100}%`;
    $('footerStatus').textContent = `已完成 ${reviewedCount()} / 24 段`;
  });
  return saves;
}
async function persistClip(reviewed) {
  if (!draft || readOnly()) return;
  clearTimeout(saveTimer);
  setSaveState('保存中…');
  try { await postClip(reviewed); }
  catch (error) { setSaveState('保存失败'); toast(error.message, true); throw error; }
}
function scheduleSave() {
  if (readOnly()) return;
  setSaveState('尚未保存');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => persistClip(false).catch(() => {}), 900);
}
async function saveMeta() {
  clearTimeout(metaTimer);
  metaTimer = null;
  try {
    const payload = JSON.stringify({ annotator: ui.annotator.value.trim(), rules: ui.rules.value });
    saves = saves.catch(() => {}).then(async () => {
      const result = await api('/api/meta', { method: 'POST', body: payload });
      session.state = result.state;
      setSaveState('已保存');
    });
    await saves;
  } catch (error) { setSaveState('保存失败'); toast(error.message, true); throw error; }
}
function scheduleMeta() {
  setSaveState('尚未保存');
  clearTimeout(metaTimer);
  metaTimer = setTimeout(() => saveMeta().catch(() => {}), 900);
}

async function completeAndNext() {
  if (!draft.segments.length && !draft.no_speech_confirmed) {
    toast('请标出可闻语音，或勾选“确认本段没有可闻语音”', true); return;
  }
  try { await persistClip(true); } catch (_) { return; }
  if (metaTimer) { try { await saveMeta(); } catch (_) { return; } }
  draft.reviewed = true;
  if (currentIndex === session.clips.length - 1) {
    render(); toast('24 段已完成。请导出 CSV 记录。'); return;
  }
  const next = session.clips[currentIndex + 1];
  if (next.phase === 'audit' && !session.state.rules_frozen) {
    render(); toast('D 阶段完成：写好并固定右侧判断规则，再开始 A 阶段。'); return;
  }
  await openClip(currentIndex + 1);
}

async function freezeRules() {
  try {
    await saveMeta();
    const result = await api('/api/freeze', { method: 'POST', body: '{}' });
    session.state = result.state;
    await openClip(session.clips.findIndex(c => c.phase === 'audit'));
    toast('规则已固定，开始 A 阶段');
  } catch (error) { toast(error.message, true); }
}

async function exportRecords() {
  try {
    clearTimeout(saveTimer); clearTimeout(metaTimer);
    if (draft && !readOnly()) await persistClip(false);
    if (!session.state.rules_frozen) await saveMeta();
    const response = await fetch('/api/export');
    if (!response.ok) throw new Error('导出失败');
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'human_review_annotations.zip';
    document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast('已导出 ZIP；服务器也保存了两份 CSV 和规则文本');
  } catch (error) { toast(error.message, true); }
}

ui.canvas.addEventListener('pointerdown', event => {
  if (readOnly()) return;
  dragStart = waveTime(event); selection = [dragStart, dragStart];
  ui.canvas.setPointerCapture(event.pointerId); drawWave();
});
ui.canvas.addEventListener('pointermove', event => {
  if (dragStart === null) return;
  const end = waveTime(event);
  selection = [Math.min(dragStart, end), Math.max(dragStart, end)];
  updateSelectionText(); drawWave();
});
ui.canvas.addEventListener('pointerup', event => {
  if (dragStart === null) return;
  const end = waveTime(event);
  if (Math.abs(end - dragStart) < .02) {
    ui.audio.currentTime = end; selection = null;
  } else selection = [Math.min(dragStart, end), Math.max(dragStart, end)];
  dragStart = null; updateSelectionText(); drawWave();
});
$('addSpeech').addEventListener('click', () => addSelection('speech'));
$('addUncertain').addEventListener('click', () => addSelection('uncertain'));
$('playPause').addEventListener('click', async () => {
  if (ui.audio.paused) { try { await ui.audio.play(); } catch (error) { toast(error.message, true); } }
  else ui.audio.pause();
});
ui.audio.addEventListener('play', () => $('playPause').textContent = 'Ⅱ');
ui.audio.addEventListener('pause', () => $('playPause').textContent = '▶');
ui.audio.addEventListener('timeupdate', () => {
  $('seek').value = String(ui.audio.currentTime);
  $('timeReadout').textContent = `${formatTime(ui.audio.currentTime)} / ${formatTime(clip().duration_seconds)}`;
  drawWave();
});
$('seek').addEventListener('input', event => { ui.audio.currentTime = Number(event.target.value); drawWave(); });
$('speed').addEventListener('change', event => ui.audio.playbackRate = Number(event.target.value));
$('noSpeech').addEventListener('change', event => {
  if (event.target.checked && draft.segments.length) {
    event.target.checked = false; toast('请先删除已有标注，再确认无语音', true); return;
  }
  draft.no_speech_confirmed = event.target.checked; scheduleSave();
});
$('difficult').addEventListener('change', event => { draft.difficult_to_judge = event.target.checked; scheduleSave(); });
$('clipNotes').addEventListener('input', event => { draft.notes = event.target.value; scheduleSave(); });
ui.rules.addEventListener('input', scheduleMeta);
ui.annotator.addEventListener('change', () => saveMeta().catch(() => {}));
$('freeze').addEventListener('click', freezeRules);
$('completeNext').addEventListener('click', completeAndNext);
$('previous').addEventListener('click', () => openClip(currentIndex - 1).catch(error => toast(error.message, true)));
$('next').addEventListener('click', () => openClip(currentIndex + 1).catch(error => toast(error.message, true)));
$('exportTop').addEventListener('click', exportRecords);
$('exportBottom').addEventListener('click', exportRecords);
document.addEventListener('keydown', event => {
  if (event.code === 'Space' && !['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(document.activeElement?.tagName)) {
    event.preventDefault(); $('playPause').click();
  }
});
new ResizeObserver(drawWave).observe(ui.waveWrap);

(async () => {
  try {
    session = await api('/api/session');
    const firstUnreviewed = session.clips.findIndex(c => !session.state.clips[c.clip_id]?.reviewed && (c.phase === 'development' || session.state.rules_frozen));
    await openClip(firstUnreviewed >= 0 ? firstUnreviewed : session.state.rules_frozen ? 23 : 7);
    setSaveState('已保存');
  } catch (error) {
    $('clipTitle').textContent = '无法载入标注任务';
    toast(error.message, true);
  }
})();
