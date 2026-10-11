import { batch, createEffect, createMemo, createSignal, For, Show, on, onCleanup, onMount } from 'solid-js';
import type { Artifact, Compilation, TraceResult } from 'blaten/trace';
import { Code } from './Code';
import { kernelPaths, kernelSources, type KernelSource } from './trace-model';
import { IRGraph, kernelNodes, type GraphNode } from './IRGraph';
import type { KernelIR } from 'blaten/ir';

interface Node { id: string; op: string; shape: number[]; dtype: string; inputs: string[]; origins?: { kernel: string; instruction: number }[] }
interface Dispatch { id: string; output: string; shader: string; fusedNodes: string[]; workgroupCount: number[]; workgroupSize: number[]; resources: { name: string; node: string; access: string }[] }
interface Kernel { id: string; name?: string; kind: string; ir?: KernelIR; source?: { file: string; artifact?: string; start: number; end: number } }
const payload = <T,>(artifact?: Artifact): T | undefined => artifact?.content.type === 'json' ? artifact.content.value as unknown as T : undefined;

/** The viewer depends only on portable trace data, never the backend. */
export function TraceView(props: { result?: TraceResult; initialSource: string; file: string }) {
  const [compilation, setCompilation] = createSignal('');
  const [stage, setStage] = createSignal('source');
  const [selected, setSelected] = createSignal('');
  let pendingInstruction: string | undefined;
  let pendingTensor: string | undefined;
  const [sourceSpans, setSourceSpans] = createSignal<readonly { start: number; end: number }[]>([]);
  let sourceView: HTMLPreElement | undefined;
  const [focus, setFocus] = createSignal<KernelSource>();
  const [menu, setMenu] = createSignal<{ kernel: KernelSource; candidates: Compilation[]; x: number; y: number; trigger: HTMLElement }>();
  let menuView: HTMLDivElement | undefined;
  const record = createMemo(() => props.result?.compilations.find(c => c.id === compilation()));
  const choose = (kernel: KernelSource, c: Compilation) => {
    setFocus(kernel); setCompilation(c.id); setSourceSpans([]); setSelected(''); setStage('kernel'); setMenu(undefined);
  };
  const clearKernel = () => {
    setFocus(undefined); setCompilation(''); setSourceSpans([]); setSelected(''); setMenu(undefined); setStage('source');
  };
  const selectKernel = (kernel: KernelSource, event: MouseEvent) => {
    const candidates = (props.result?.compilations || []).filter(c => kernelPaths(c, kernel).length)
      .sort((a, b) => Math.min(...kernelPaths(a, kernel).map(p => p.length)) - Math.min(...kernelPaths(b, kernel).map(p => p.length)));
    if (candidates.length === 1) { choose(kernel, candidates[0]!); return; }
    if (!candidates.length) return;
    const trigger = event.currentTarget as HTMLElement, bounds = trigger.getBoundingClientRect();
    setMenu({ kernel, candidates, x: event.detail ? event.clientX : bounds.left, y: event.detail ? event.clientY : bounds.top + 24, trigger });
  };
  onMount(() => {
    const dismiss = (event: PointerEvent) => { if (!menuView?.contains(event.target as globalThis.Node)) setMenu(undefined); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape' && menu()) { menu()!.trigger.focus(); setMenu(undefined); } };
    document.addEventListener('pointerdown', dismiss); document.addEventListener('keydown', escape);
    onCleanup(() => { document.removeEventListener('pointerdown', dismiss); document.removeEventListener('keydown', escape); });
  });
  createEffect(() => {
    const value = menu(); if (!value) return;
    queueMicrotask(() => { if (menu() === value) menuView?.querySelector('button')?.focus(); });
    const frame = requestAnimationFrame(() => {
      if (!menuView) return;
      const bounds = menuView.getBoundingClientRect();
      menuView.style.left = Math.max(8, Math.min(value.x, innerWidth - bounds.width - 8)) + 'px';
      menuView.style.top = Math.max(8, Math.min(value.y, innerHeight - bounds.height - 8)) + 'px';
    });
    onCleanup(() => cancelAnimationFrame(frame));
  });
  const graph = createMemo(() => payload<{ nodes: Node[]; inputs: string[]; outputs: string[] }>(record()?.artifacts.find(a => a.kind === 'blaten.tensor-ir')));
  const plan = createMemo(() => payload<{ dispatches: Dispatch[]; workspaceBytes: number }>(record()?.artifacts.find(a => a.kind === 'typegpu.execution-plan')));
  const kernels = createMemo(() => payload<{ kernels: Kernel[] }>(record()?.artifacts.find(a => a.kind === 'blaten.kernel-ir'))?.kernels.filter(k => k.kind === 'kernel') || []);
  const dispatch = createMemo(() => plan()?.dispatches.find(d => d.id === selected()));
  const node = createMemo(() => graph()?.nodes.find(n => n.id === selected()));
  const sourceKernel = createMemo(() => kernels().find(k => focus() && k.source?.file === focus()?.file && k.source?.start === focus()?.start && k.source?.end === focus()?.end) || kernels().find(k => k.name === 'loss') || kernels()[0]);
  const instruction = createMemo(() => {
    const ir = sourceKernel()?.ir; if (!ir || !/^i\d+$/.test(selected())) return;
    const index = Number(selected().slice(1));
    return ir.instructions[index] ? { id: selected(), value: ir.instructions[index] } : undefined;
  });
  const label = (id: string) => { const n = graph()?.nodes.find(n => n.id === id); return n ? `${id} ${n.op} [${n.shape.join('×')}]` : id; };
  const planNodes = createMemo<GraphNode[]>(() => {
    const dispatches = plan()?.dispatches || [];
    const producers = new Map(dispatches.map(d => [d.output, d.id]));
    const buffers = [...new Set(dispatches.flatMap(d => d.resources.filter(r => r.access === 'read' && !producers.has(r.node)).map(r => r.node)))];
    return [
      ...buffers.map(id => ({ id: `buffer:${id}`, title: graph()?.nodes.find(n => n.id === id)?.op || 'buffer', detail: graph()?.inputs.includes(id) ? 'Application input' : 'Constant / prepared buffer', inputs: [], input: true })),
      ...dispatches.map(d => ({ id: d.id, title: d.shader, detail: `→ ${label(d.output)}`, inputs: d.resources.filter(r => r.access === 'read').map(r => ({ id: producers.get(r.node) || `buffer:${r.node}`, label: r.node })), output: graph()?.outputs.includes(d.output) })),
    ];
  });
  const shader = createMemo(() => record()?.artifacts.find(a => a.id === dispatch()?.shader && a.content.type === 'text' && a.content.language === 'wgsl'));
  const source = createMemo(() => props.result?.sources.find(a => a.id === sourceKernel()?.source?.artifact) || props.result?.sources.find(a => a.location?.file === props.file));
  const sourceText = createMemo(() => source()?.content.type === 'text' ? source()!.content.value as string : props.initialSource);
  const locations = createMemo(() => {
    const text = sourceText();
    const spans = [...(instruction() ? sourceKernel()?.ir?.spans?.[Number(selected().slice(1))] || [] : [])]
      .filter(s => s.start >= 0 && s.end > s.start && s.end <= text.length)
      .sort((a, b) => a.start - b.start || a.end - b.end);
    const groups: { start: number; end: number; spans: { start: number; end: number }[] }[] = [];
    for (const span of spans) {
      const start = text.lastIndexOf('\n', span.start - 1) + 1;
      const newline = text.indexOf('\n', span.end - 1);
      const end = newline < 0 ? text.length : newline;
      const previous = groups.at(-1);
      // Merge snippets sharing a source line, including multiline spans.
      if (previous && start <= previous.end) {
        previous.end = Math.max(previous.end, end);
        previous.spans.push(span);
      } else groups.push({ start, end, spans: [span] });
    }
    return groups.map(group => ({
      start: group.spans[0]!.start,
      line: text.slice(0, group.start).split('\n').length,
      column: group.spans[0]!.start - group.start + 1,
      text: text.slice(group.start, group.end),
      highlights: group.spans.map(span => ({ start: span.start - group.start, end: span.end - group.start })),
    }));
  });
  const selectInstruction = (id: string) => {
    setSelected(id);
    setSourceSpans(sourceKernel()?.ir?.spans?.[Number(id.slice(1))] || []);
  };
  const showOrigin = (origin: { kernel: string; instruction: number }) => {
    const kernel = kernels().find(k => k.id === origin.kernel);
    if (!kernel?.source || !kernel.ir?.instructions[origin.instruction]) return;
    batch(() => {
      pendingInstruction = `i${origin.instruction}`;
      setFocus({ ...kernel.source!, name: kernel.name || kernel.id });
      setStage('kernel');
    });
  };
  const showTensor = (id: string) => {
    if (!graph()?.nodes.some(n => n.id === id)) return;
    pendingTensor = id;
    setStage('tensor');
  };
  const dispatchNodes = createMemo(() => {
    const d = dispatch();
    return d ? [...new Set([d.output, ...d.fusedNodes, ...d.resources.map(r => r.node)])]
      .filter(id => graph()?.nodes.some(n => n.id === id)) : [];
  });
  createEffect(on(() => [record()?.id, stage(), focus()], () => {
    const initial = pendingInstruction || 'i0';
    pendingInstruction = undefined;
    setSelected(stage() === 'kernel' ? sourceKernel()?.ir?.instructions.length ? initial : ''
      : stage() === 'tensor' ? pendingTensor || graph()?.nodes[0]?.id || ''
      : stage() === 'plan' ? plan()?.dispatches[0]?.id || '' : '');
    pendingTensor = undefined;
    if (stage() === 'kernel') setSourceSpans(sourceKernel()?.ir?.spans?.[Number(initial.slice(1))] || []);
  }));
  createEffect(() => {
    const spans = sourceSpans();
    if (stage() !== 'source' || !spans.length) return;
    const frame = requestAnimationFrame(() => sourceView?.querySelector('.highlighted')?.scrollIntoView({ block: 'center' }));
    onCleanup(() => cancelAnimationFrame(frame));
  });
  const selectStage = (id: string) => { setStage(id); };
  return <section class="panel trace-panel">
    <div class="section-heading"><h2>Model</h2><Show when={focus()}>{kernel => <div class="trace-actions"><div class="selected-kernel"><Show when={record()}>{c => <For each={kernelPaths(c(), kernel())}>{path => <span class="selected-compilation"><For each={path}>{(step, index) => <><Show when={index() > 0}> → </Show><Show when={kernels().find(k => k.id === step.id)?.source?.start === kernel().start && kernels().find(k => k.id === step.id)?.source?.end === kernel().end} fallback={step.name}><strong>{step.name}</strong></Show></>}</For></span>}</For>}</Show></div><button class="close-button" aria-label="Clear kernel selection" onClick={clearKernel}>×</button></div>}</Show></div>
    <nav class="tabs compilation-tabs" aria-label="Compilation stages"><For each={[['source', 'TS'], ['kernel', 'Kernel IR'], ['tensor', 'Tensor IR'], ['plan', 'WGSL']]}>{([id, title], index) => <><Show when={index() > 0}><span class="stage-arrow" aria-hidden="true">→</span></Show><button classList={{ active: stage() === id }} disabled={id !== 'source' && !record()?.stages.some(s => s.id === id)} onClick={() => selectStage(id)}>{title}</button></>}</For></nav>
    <Show when={stage() === 'source'}><pre ref={sourceView} class="model-code trace-source"><Code source={sourceText()} language="typescript" kernels={kernelSources(props.result).filter(k => k.file === props.file)} selected={focus()} onSelect={selectKernel} highlights={sourceSpans()} /></pre></Show>
    <Show when={menu()}>{m => <div ref={menuView} class="kernel-menu" role="menu" aria-label={`${m().kernel.name} compilation choices`} style={{ left: `${m().x}px`, top: `${m().y}px` }} onKeyDown={event => {
      if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault(); const buttons = Array.from(menuView!.querySelectorAll('button'));
      const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (current + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next]?.focus();
    }}><For each={m().candidates}>{c => <button role="menuitem" onClick={() => choose(m().kernel, c)}><For each={kernelPaths(c, m().kernel)}>{path => <span>{path.map(step => step.name).join(' → ')}</span>}</For></button>}</For></div>}</Show>
    <Show when={record()}>{c => <>
      <Show when={stage() !== 'source'}>
      <div class="trace-workspace" classList={{ 'has-detail': !!selected() }}>
        <div class="trace-graph">
          <Show when={stage() === 'plan'}>
            <Show when={plan()?.dispatches.length} fallback={<p class="hint">No GPU dispatches; input buffers are copied.</p>}>
              <IRGraph name="Execution plan" nodes={planNodes()} selected={selected()} onSelect={setSelected} />
            </Show>
          </Show>
          <Show when={stage() === 'tensor' && graph()}>{g =>
            <IRGraph name="Tensor IR" nodes={g().nodes.map(n => ({ id: n.id, title: n.op, detail: `[${n.shape.join(' × ')}] ${n.dtype}`, inputs: n.inputs.map((id, i) => ({ id, label: `input ${i}` })), input: g().inputs.includes(n.id), output: g().outputs.includes(n.id) }))} selected={selected()} onSelect={setSelected} />
          }</Show>
          <Show when={stage() === 'kernel' && sourceKernel()?.ir}>{ir =>
            <IRGraph name="Kernel IR" nodes={kernelNodes(ir())} selected={selected()} onSelect={selectInstruction} />
          }</Show>
        </div>
        <Show when={selected()}>
          <aside class="node-detail" aria-label="Selected node details">
            <div class="detail-heading"><h3>{selected()}</h3><button class="close-button" aria-label="Close node details" onClick={() => setSelected('')}>×</button></div>
            <Show when={stage() === 'kernel' && instruction()}>{i => <>
              <div class="code-heading"><span>{sourceKernel()?.name} · {i().id}</span></div>
              <Show when={locations().length} fallback={<p class="hint">No TS source locations recorded.</p>}>
                <h3>TS · {sourceKernel()?.source?.file}</h3>
                <For each={locations()}>{location => <div class="source-location" data-span-start={location.start}>
                  <div class="code-heading"><span>{location.line}:{location.column}</span></div>
                  <pre><Code source={location.text} language="typescript" highlights={location.highlights} /></pre>
                </div>}</For>
              </Show>
              <pre class="instruction-detail"><Code source={JSON.stringify(i().value, null, 2)} language="json" /></pre>
            </>}</Show>
            <Show when={stage() === 'tensor' && node()}>{n => <>
              <div class="code-heading"><span>{label(n().id)}</span></div>
              <Show when={n().origins?.length}>
                <h3>Kernel IR</h3>
                <div class="kernel-origins"><For each={n().origins}>{origin => {
                  const kernel = kernels().find(k => k.id === origin.kernel);
                  return <button disabled={!kernel?.source} onClick={() => showOrigin(origin)}>{kernel?.name || origin.kernel} · i{origin.instruction}</button>;
                }}</For></div>
              </Show>
              <pre class="tensor-detail"><Code source={JSON.stringify(n(), null, 2)} language="json" /></pre>
            </>}</Show>
            <Show when={stage() === 'plan' && dispatch()}>{d => <>
              <div class="code-heading"><span>{d().shader} → {label(d().output)}</span><span>{d().workgroupCount.join(' × ')} groups · {d().workgroupSize.join(' × ')} threads / group</span></div>
              <Show when={dispatchNodes().length}>
                <h3>Tensor IR</h3>
                <div class="kernel-origins tensor-origins"><For each={dispatchNodes()}>{id =>
                  <button title={[
                    ...(id === d().output ? ['Output'] : []),
                    ...(d().fusedNodes.includes(id) ? ['Fused operation'] : []),
                    ...d().resources.filter(r => r.node === id).map(r => `${r.name} · ${r.access}`),
                  ].join(' · ')} onClick={() => showTensor(id)}>{label(id)}</button>
                }</For></div>
              </Show>
              <Show when={shader()} fallback={<p class="hint">No shader code recorded for this dispatch.</p>}>{a => <pre class="source-code shader-code"><Code source={a().content.value as string} language="wgsl" /></pre>}</Show>
              <pre><Code source={JSON.stringify(d(), null, 2)} language="json" /></pre>
            </>}</Show>
            <Show when={stage() === 'plan' && selected().startsWith('buffer:')}>
              <div class="code-heading"><span>{label(selected().slice(7))} · Input buffer</span></div>
              <div class="kernel-origins tensor-origins"><button onClick={() => showTensor(selected().slice(7))}>{label(selected().slice(7))}</button></div>
              <pre><Code source={JSON.stringify(graph()?.nodes.find(n => n.id === selected().slice(7)), null, 2)} language="json" /></pre>
            </Show>
          </aside>
        </Show>
      </div>
      </Show>
      <Show when={c().diagnostics.length}><pre role="alert">{JSON.stringify(c().diagnostics, null, 2)}</pre></Show>
    </>}</Show>
  </section>;
}
