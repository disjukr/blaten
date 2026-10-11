import { createEffect, createMemo, createSignal, For, onCleanup, onMount, Show } from 'solid-js';
import { loadSplit } from './data';

function Sample(props: { pixels: Uint8Array; index: number; label: number; selected?: boolean; onSelect?: () => void }) {
  let canvas!: HTMLCanvasElement;
  createEffect(() => {
    const context = canvas.getContext('2d')!;
    const image = context.createImageData(28, 28);
    for (let p = 0; p < 784; p++) {
      const value = props.pixels[props.index * 784 + p]!;
      image.data.set([value, value, value, 255], p * 4);
    }
    context.putImageData(image, 0, 0);
  });
  return <button class="dataset-sample" classList={{ selected: props.selected }} aria-label={`Select training sample ${props.index + 1}, label ${props.label}`} aria-pressed={props.selected || false} onClick={props.onSelect}>
    <canvas ref={canvas} width="28" height="28" aria-label={`Training sample ${props.index + 1}, label ${props.label}`} />
    <span class="dataset-caption"><strong>{props.label}</strong><span>#{props.index + 1}</span></span>
  </button>;
}

export function DatasetView(props: { trainSize: number; onSelect: (pixels: Float32Array) => void }) {
  const [data, setData] = createSignal<Awaited<ReturnType<typeof loadSplit>>>();
  const [error, setError] = createSignal('');
  const [digit, setDigit] = createSignal('all');
  const [scrollTop, setScrollTop] = createSignal(0);
  const [columns, setColumns] = createSignal(1);
  const [viewportHeight, setViewportHeight] = createSignal(420);
  const [selected, setSelected] = createSignal<number>();
  let viewport: HTMLDivElement | undefined;
  const controller = new AbortController();
  onCleanup(() => controller.abort());
  const load = async () => {
    setError('');
    try { setData(await loadSplit('train', controller.signal)); }
    catch (error) { if (!controller.signal.aborted) setError(error instanceof Error ? error.message : String(error)); }
  };
  onMount(() => { void load(); });
  const indices = createMemo(() => {
    const split = data();
    if (!split) return [];
    const count = Math.min(split.labels.length, Math.max(0, Math.floor(props.trainSize) || 0));
    return Array.from({ length: count }, (_, i) => i).filter(i => digit() === 'all' || split.labels[i] === Number(digit()));
  });
  const rowHeight = 112;
  const rows = () => Math.ceil(indices().length / columns());
  const startRow = () => Math.max(0, Math.min(rows() - 1, Math.floor(scrollTop() / rowHeight)) - 2);
  const endRow = () => Math.min(rows(), Math.ceil((scrollTop() + viewportHeight()) / rowHeight) + 2);
  createEffect(() => { indices(); setScrollTop(0); if (viewport) viewport.scrollTop = 0; });
  createEffect(() => {
    if (!data()) return;
    let observer: ResizeObserver | undefined;
    const frame = requestAnimationFrame(() => {
      if (!viewport) return;
      const measure = () => {
        setColumns(Math.max(1, Math.floor((viewport!.clientWidth - 4 + 12) / 88)));
        setViewportHeight(viewport!.clientHeight);
        setScrollTop(viewport!.scrollTop);
      };
      measure(); observer = new ResizeObserver(measure); observer.observe(viewport);
    });
    onCleanup(() => { cancelAnimationFrame(frame); observer?.disconnect(); });
  });
  const visible = () => indices().slice(startRow() * columns(), endRow() * columns());
  const select = (index: number) => {
    setSelected(index);
    props.onSelect(Float32Array.from(data()!.pixels.subarray(index * 784, (index + 1) * 784), value => value / 255));
  };
  return <section class="panel dataset-panel">
    <div class="section-heading"><h2>Training dataset</h2>
      <label class="dataset-filter">Label <select value={digit()} onChange={e => setDigit(e.currentTarget.value)}>
        <option value="all">All digits</option><For each={Array.from({ length: 10 }, (_, i) => i)}>{n => <option value={n}>{n}</option>}</For>
      </select></label>
    </div>
    <Show when={data()} fallback={<Show when={error()} fallback={<p class="hint">Loading training images…</p>}>
      <p class="error" role="alert">{error()}</p><button onClick={() => void load()}>Retry</button>
    </Show>}>{split => <>
      <div ref={viewport} class="dataset-scroll" onScroll={e => setScrollTop(e.currentTarget.scrollTop)} tabindex="0" role="region" aria-label="Training images">
        <div class="dataset-spacer" style={{ height: `${Math.max(0, rows() * rowHeight - 12)}px` }}>
          <div class="dataset-grid" style={{ top: `${startRow() * rowHeight}px`, 'grid-template-columns': `repeat(${columns()},minmax(0,1fr))` }}><For each={visible()}>{index => <Sample pixels={split().pixels} index={index} label={split().labels[index]!} selected={selected() === index} onSelect={() => select(index)} />}</For></div>
        </div>
        <Show when={!indices().length}><p class="hint">No samples</p></Show>
      </div>
      <p class="hint">{indices().length.toLocaleString()} images · 28 × 28</p>
    </>}</Show>
  </section>;
}
