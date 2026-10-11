import { createEffect, createSignal, For, onMount, onCleanup, Show } from 'solid-js';

export function DigitTest(props: { disabled: boolean; version: unknown; predict: (image: Float32Array) => Promise<number[]>; input?: Float32Array }) {
  let canvas!: HTMLCanvasElement, preview!: HTMLCanvasElement;
  let drawing = false;
  const [image, setImage] = createSignal(new Float32Array(784));
  const [scores, setScores] = createSignal<number[]>();
  const [error, setError] = createSignal('');
  let revision = 0, running = false, disposed = false, drawFrame = 0;
  let queued: Float32Array | undefined, timer: ReturnType<typeof setTimeout> | undefined;
  const clear = () => {
    canvas.getContext('2d')!.clearRect(0, 0, 224, 224);
    setImage(new Float32Array(784)); setScores(undefined); setError('');
  };
  onMount(clear);
  createEffect(() => {
    const pixels = props.input;
    if (!pixels) return;
    drawing = false;
    if (drawFrame) cancelAnimationFrame(drawFrame);
    drawFrame = 0;
    const small = document.createElement('canvas'); small.width = small.height = 28;
    const context = small.getContext('2d')!, data = context.createImageData(28, 28);
    pixels.forEach((value, i) => {
      data.data[i * 4] = data.data[i * 4 + 1] = data.data[i * 4 + 2] = 255;
      data.data[i * 4 + 3] = Math.round(value * 255);
    });
    context.putImageData(data, 0, 0);
    const pad = canvas.getContext('2d')!;
    pad.clearRect(0, 0, 224, 224); pad.imageSmoothingEnabled = false;
    pad.drawImage(small, 0, 0, 224, 224);
    setScores(undefined); setError(''); setImage(pixels.slice());
  });
  createEffect(() => {
    const pixels = image(), context = preview.getContext('2d')!, data = context.createImageData(28, 28);
    pixels.forEach((value, i) => { data.data[i * 4] = data.data[i * 4 + 1] = data.data[i * 4 + 2] = Math.round(value * 255); data.data[i * 4 + 3] = 255; });
    context.putImageData(data, 0, 0);
  });
  const normalize = () => {
    const pixels = canvas.getContext('2d')!.getImageData(0, 0, 224, 224).data;
    let left = 224, right = -1, top = 224, bottom = -1;
    for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) {
      if (pixels[(y * 224 + x) * 4 + 3]! > 16) { left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y); }
    }
    const result = new Float32Array(784);
    if (right < left) return result;
    const small = document.createElement('canvas'); small.width = small.height = 28;
    const context = small.getContext('2d')!, width = right - left + 1, height = bottom - top + 1, scale = 20 / Math.max(width, height);
    context.drawImage(canvas, left, top, width, height, (28 - width * scale) / 2, (28 - height * scale) / 2, width * scale, height * scale);
    const data = context.getImageData(0, 0, 28, 28).data;
    let mass = 0, centerX = 0, centerY = 0;
    for (let i = 0; i < 784; i++) { const value = data[i * 4 + 3]! / 255; result[i] = value; mass += value; centerX += (i % 28) * value; centerY += Math.floor(i / 28) * value; }
    if (!mass) return result;
    const dx = Math.round(13.5 - centerX / mass), dy = Math.round(13.5 - centerY / mass), centered = new Float32Array(784);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      if (x + dx >= 0 && x + dx < 28 && y + dy >= 0 && y + dy < 28) centered[(y + dy) * 28 + x + dx] = result[y * 28 + x]!;
    }
    return centered;
  };
  const point = (event: PointerEvent) => { const bounds = canvas.getBoundingClientRect(); return [(event.clientX - bounds.left) * 224 / bounds.width, (event.clientY - bounds.top) * 224 / bounds.height] as const; };
  const refresh = () => {
    if (drawFrame) return;
    drawFrame = requestAnimationFrame(() => { drawFrame = 0; setImage(normalize()); });
  };
  const finish = () => { if (drawing) { drawing = false; if (drawFrame) cancelAnimationFrame(drawFrame); drawFrame = 0; setImage(normalize()); } };
  const schedule = () => {
    if (running || timer || !queued || disposed) return;
    timer = setTimeout(() => { timer = undefined; void test(); }, 75);
  };
  const test = async () => {
    if (!queued || props.disabled || disposed) return;
    const pixels = queued, current = revision; queued = undefined; running = true; setError('');
    try {
      const result = await props.predict(pixels);
      if (!disposed && current === revision) setScores(result);
    } catch (e) { if (!disposed && current === revision) setError(e instanceof Error ? e.message : String(e)); }
    finally { running = false; schedule(); }
  };
  createEffect(() => {
    const pixels = image(), disabled = props.disabled; props.version;
    revision++;
    if (disabled || !pixels.some(value => value > 0)) { queued = undefined; setScores(undefined); return; }
    queued = pixels; schedule();
  });
  onCleanup(() => { disposed = true; revision++; if (timer) clearTimeout(timer); if (drawFrame) cancelAnimationFrame(drawFrame); });
  return <section class="panel digit-test">
    <div class="section-heading"><h2>Test the model</h2></div>
    <div class="digit-test-layout"><div>
      <p class="hint">Draw a digit or select a dataset image.</p>
      <canvas ref={canvas} class="drawing-pad" width="224" height="224" aria-label="Digit drawing pad"
        onPointerDown={event => {
          event.preventDefault(); drawing = true; canvas.setPointerCapture(event.pointerId); setScores(undefined);
          const context = canvas.getContext('2d')!, [x, y] = point(event);
          context.strokeStyle = context.fillStyle = '#fff'; context.lineWidth = 16; context.lineCap = context.lineJoin = 'round';
          context.beginPath(); context.arc(x, y, 8, 0, Math.PI * 2); context.fill(); context.beginPath(); context.moveTo(x, y); refresh();
        }}
        onPointerMove={event => { if (!drawing) return; const [x, y] = point(event), context = canvas.getContext('2d')!; context.lineTo(x, y); context.stroke(); refresh(); }}
        onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish} />
      <div class="actions"><div class="digit-input"><canvas ref={preview} width="28" height="28" role="img" aria-label="Normalized 28 by 28 model input" /><span class="hint">28 × 28</span></div><button onClick={clear}>Clear</button></div>
      <Show when={error()}><p class="error" role="alert">{error()}</p></Show>
    </div><div class="digit-results" aria-live="polite">
      <Show when={props.disabled}><p class="hint">Train the model to test your drawing.</p></Show>
      <Show when={scores()}>{values => <>
        <p class="digit-prediction">Prediction: <strong>{values().indexOf(Math.max(...values()))}</strong></p>
        <div class="digit-scores"><For each={values()}>{(score, digit) => <div class="digit-score" classList={{ best: score === Math.max(...values()) }}><span>{digit()}</span><meter min="0" max="1" value={score} aria-label={`Digit ${digit()} score`} /><span>{(score * 100).toFixed(1)}%</span></div>}</For></div>
      </>}</Show>
    </div></div>
  </section>;
}
