import { render } from 'solid-js/web';
import { createEffect, createMemo, createSignal, For, onCleanup, onMount, Show } from 'solid-js';
import type { TraceResult } from 'blaten/trace';
import { createTrainer, type Measurement, type Options, type Weights } from './train';
import { DigitTest } from './DigitTest';
import { DatasetView } from './DatasetView';
import { TraceView } from './TraceView';
import modelSource from './model.ts?raw';
import './style.css';

function Heatmap(props: { values: number[]; width: number; height: number; label: string; scale: number }) {
  let canvas!: HTMLCanvasElement;
  createEffect(() => {
    const data = props.values, scale = props.scale || 1;
    const ctx = canvas.getContext('2d')!;
    const image = ctx.createImageData(props.width, props.height);
    data.forEach((value, i) => {
      const t = Math.max(-1, Math.min(1, value / scale)), amount = Math.abs(t);
      const target = t >= 0 ? [244, 128, 101] : [102, 158, 247];
      for (let c = 0; c < 3; c++) image.data[i * 4 + c] = Math.round(26 + (target[c]! - 26) * amount);
      image.data[i * 4 + 3] = 255;
    });
    ctx.putImageData(image, 0, 0);
  });
  return <canvas ref={canvas} width={props.width} height={props.height} role="img" aria-label={props.label} />;
}

function History(props: { points: Measurement[] }) {
  const chart = (metric: 'loss' | 'accuracy') => {
    const values = () => props.points.filter(p => p[metric] !== undefined);
    const max = () => metric === 'accuracy' ? 1 : Math.max(1, ...values().map(p => p.loss!));
    const x = (p: Measurement) => 36 + p.epoch / Math.max(1, ...props.points.map(p => p.epoch)) * 264;
    const y = (p: Measurement) => 110 - p[metric]! / max() * 82;
    return <div><h3>{metric === 'loss' ? 'Training loss' : 'Test accuracy'}</h3><svg viewBox="0 0 320 142" role="img" aria-label={metric === 'loss' ? 'Mean loss per epoch' : 'Test accuracy per epoch'}>
      <path class="axis" d="M36 22V110H300" /><text x="5" y="27">{metric === 'accuracy' ? '100%' : max().toFixed(1)}</text><text x="20" y="113">0</text><text x="260" y="136">Epoch</text>
      <polyline class="chart-line" points={values().map(p => `${x(p)},${y(p)}`).join(' ')} />
      <For each={values()}>{p => <g><circle cx={x(p)} cy={y(p)} r="3.5" /><text x={x(p)} y="126" text-anchor="middle">{p.epoch}</text><title>{metric === 'accuracy' ? (p.accuracy * 100).toFixed(2) + '%' : p.loss?.toFixed(4)}</title></g>}</For>
    </svg></div>;
  };
  return <div class="charts">{chart('loss')}{chart('accuracy')}</div>;
}

function App() {
  const [options, setOptions] = createSignal<Options>({ epochs: 3, trainSize: 12000, testSize: 2000, batchSize: 64, learningRate: 0.1 });
  const [busy, setBusy] = createSignal(false), [status, setStatus] = createSignal('Ready'), [progress, setProgress] = createSignal(0);
  const [error, setError] = createSignal(''), [device, setDevice] = createSignal('');
  const [trace, setTrace] = createSignal<TraceResult>(), [weights, setWeights] = createSignal<Weights>(), [initial, setInitial] = createSignal<Weights>();
  const [history, setHistory] = createSignal<Measurement[]>([]), [mode, setMode] = createSignal('current');
  const [resetWeights, setResetWeights] = createSignal(false);
  const [trained, setTrained] = createSignal(false);
  const [testing, setTesting] = createSignal(false);
  const [testImage, setTestImage] = createSignal<Float32Array>();
  let controller: AbortController | undefined;
  let trainer: Awaited<ReturnType<typeof createTrainer>> | undefined;
  let disposed = false;
  let inferenceQueue = Promise.resolve();
  let pendingPredictions = 0;
  onCleanup(() => { disposed = true; controller?.abort(); if (!busy() && !testing()) trainer?.dispose(); });
  onMount(() => { void start(true); });
  async function start(compileOnly: boolean) {
    if (busy() || testing()) return;
    setBusy(true); setError(''); setProgress(0);
    const reset = !compileOnly && resetWeights();
    if (reset) { setHistory([]); setInitial(undefined); }
    controller = new AbortController();
    try {
      if (!trainer) {
        setStatus('Initializing GPU and compiling model');
        trainer = await createTrainer();
      }
      controller.signal.throwIfAborted();
      await trainer.run(options(), controller.signal, update => {
        if (update.status) setStatus(update.status);
        if (update.progress !== undefined) setProgress(update.progress);
        if (update.device) setDevice(update.device);
        if (update.weights) { if (!initial()) setInitial(update.weights); setWeights(update.weights); }
        if (update.trace) setTrace(update.trace);
        if (update.measurement) setHistory(points => [...points.filter(p => p.epoch !== update.measurement!.epoch), update.measurement!]);
        if (update.trained !== undefined) setTrained(update.trained);
      }, compileOnly, reset);
    } catch (e) {
      if (controller.signal.aborted) setStatus('Stopped · Last results retained');
      else { setError(e instanceof Error ? e.message : String(e)); setStatus('Run failed'); }
    } finally { controller = undefined; if (disposed) trainer?.dispose(); setBusy(false); }
  }
  async function predictDigit(image: Float32Array) {
    if (!trainer || busy() || disposed) throw new Error('The model is busy.');
    pendingPredictions++; setTesting(true);
    const operation = inferenceQueue.then(() => {
      if (disposed) throw new Error('The model is disposed.');
      return trainer!.predictDigit(image);
    });
    inferenceQueue = operation.then(() => {}, () => {});
    try { return await operation; }
    finally { if (--pendingPredictions === 0) { if (disposed) trainer.dispose(); setTesting(false); } }
  }
  const visibleWeights = createMemo(() => Object.fromEntries((['w1', 'b1', 'w2', 'b2'] as const).map(key => {
    const current = weights()?.[key] || [], before = initial()?.[key] || [];
    return [key, mode() === 'initial' ? before : mode() === 'delta' ? current.map((v, i) => v - before[i]!) : current];
  })) as unknown as Weights);
  const colorScales = createMemo(() => Object.fromEntries((['w1', 'b1', 'w2', 'b2'] as const).map(key => [key,
    mode() === 'delta' ? Math.max(0.001, ...visibleWeights()[key].map(Math.abs)) : Math.max(0.001, ...(initial()?.[key] || []).map(Math.abs), ...(weights()?.[key] || []).map(Math.abs)),
  ])) as Record<keyof Weights, number>);
  const shown = (key: keyof Weights) => visibleWeights()[key];
  const scale = (key: keyof Weights) => colorScales()[key];
  return <main>
    <header><h1>Blaten - MNIST</h1></header>
    <TraceView result={trace()} initialSource={modelSource} file="src/model.ts" />
    <section class="panel control-panel"><div class="section-heading"><h2>Training</h2></div>
      <div class="settings"><For each={[['trainSize', 'Training images', 60000, 1], ['testSize', 'Test images', 10000, 1], ['epochs', 'Epochs', 100, 1], ['batchSize', 'Batch size', 1024, 1], ['learningRate', 'Learning rate', 10, 0.01]] as const}>{([key, label, max, step]) => <label>{label}<input type="number" min={step} max={max} step={step} value={options()[key]} disabled={busy()} onInput={e => setOptions(o => ({ ...o, [key]: e.currentTarget.valueAsNumber }))} /></label>}</For></div>
      <div class="actions"><button class="primary" disabled={busy() || testing() || !navigator.gpu} onClick={() => start(false)}>{trained() && !resetWeights() ? 'Continue training' : 'Start training'}</button><Show when={busy()}><button onClick={() => { setStatus('Stopping · Waiting for pending GPU work'); controller?.abort(); }}>Stop</button></Show><label class="reset-weights"><input type="checkbox" checked={resetWeights()} disabled={busy()} onChange={e => setResetWeights(e.currentTarget.checked)} />Reset weights</label></div>
      <Show when={!navigator.gpu}><p class="error" role="alert">WebGPU is required. Open this page in Chrome or Edge on localhost or HTTPS.</p></Show>
      <div class="status" aria-live="polite"><span classList={{ pulse: busy() }} class="status-dot" />{status()}<span>{(progress() * 100).toFixed(0)}%</span></div><progress value={progress()} max="1" aria-label="Training progress" />
      <Show when={device()}><p class="hint">GPU: {device()}</p></Show><Show when={error()}><p class="error" role="alert">{error()}</p></Show>
      <Show when={history().length}><div class="results"><div><span>Before training</span><strong>{(history()[0]!.accuracy * 100).toFixed(2)}%</strong></div><div><span>Latest test accuracy</span><strong>{(history().at(-1)!.accuracy * 100).toFixed(2)}%</strong></div><div><span>Latest loss</span><strong>{history().at(-1)?.loss?.toFixed(4) || '—'}</strong></div></div><History points={history()} /></Show>
    </section>
    <DatasetView trainSize={options().trainSize} onSelect={pixels => {
      setTestImage(pixels);
      document.querySelector('.digit-test')?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    }} />
    <DigitTest disabled={busy() || !trained()} version={weights()} predict={predictDigit} input={testImage()} />
    <section class="panel"><div class="section-heading"><h2>Weights</h2><div class="tabs compact"><For each={[['current', 'Current'], ['initial', 'Initial'], ['delta', 'Change']]}>{([id, text]) => <button classList={{ active: mode() === id }} onClick={() => setMode(id)}>{text}</button>}</For></div></div>
      <Show when={weights()} fallback={<div class="empty">Available after model preparation.</div>}>
        <div class="weight-caption"><span>w1 · 784 × 32 · 32 hidden neurons</span><span class="legend"><i class="negative" /> Negative <i class="zero" /> 0 <i class="positive" /> Positive</span></div>
        <div class="weight-grid"><For each={Array.from({ length: 32 }, (_, i) => i)}>{unit => <figure><Heatmap values={Array.from({ length: 784 }, (_, pixel) => shown('w1')[pixel * 32 + unit]!)} width={28} height={28} label={`Input weights of hidden neuron ${unit + 1}`} scale={scale('w1')} /><figcaption>{String(unit + 1).padStart(2, '0')} <span>b {shown('b1')[unit]?.toFixed(3)}</span></figcaption></figure>}</For></div>
        <div class="output-weights"><div><h3>w2 · Hidden neurons → digits</h3><Heatmap values={shown('w2')} width={10} height={32} label="Output weights: columns are digits 0–9, rows are hidden neurons 1–32" scale={scale('w2')} /><div class="digit-labels"><For each={Array.from({ length: 10 }, (_, i) => i)}>{n => <span>{n}</span>}</For></div></div><div><h3>b2 · Output bias</h3><Heatmap values={shown('b2')} width={10} height={1} label="Output bias per digit" scale={scale('b2')} /><div class="digit-labels"><For each={shown('b2')}>{n => <span>{n.toFixed(2)}</span>}</For></div><p class="hint">Current and initial: shared color scale · Change: current − initial</p></div></div>
      </Show>
    </section>
  </main>;
}

render(() => <App />, document.getElementById('root')!);
