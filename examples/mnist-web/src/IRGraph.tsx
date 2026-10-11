import dagre from '@dagrejs/dagre';
import { createEffect, createMemo, createSignal, createUniqueId, For, onCleanup, onMount } from 'solid-js';
import type { KernelIR, KernelInstruction } from 'blaten/ir';

export interface GraphNode {
  id: string; title: string; detail: string;
  inputs: { id: string; label: string }[];
  input?: boolean; output?: boolean;
}

export function kernelNodes(ir: KernelIR): GraphNode[] {
  const refs = (instruction: KernelInstruction): { id: string; label: string }[] => {
    const ref = (index: number, label: string) => ({ id: `i${index}`, label });
    switch (instruction.op) {
      case 'member': return [ref(instruction.object, 'object'), ref(instruction.key, 'key')];
      case 'call': return [ref(instruction.target, 'target'), ...instruction.args.map((n, i) => ref(n, `arg ${i}`))];
      case 'binary': return [ref(instruction.left, 'left'), ref(instruction.right, 'right')];
      case 'neg': return [ref(instruction.value, 'value')];
      case 'array': return instruction.values.map((n, i) => ref(n, String(i)));
      case 'object': return Object.entries(instruction.fields).map(([key, n]) => ref(n, key));
      default: return [];
    }
  };
  return ir.instructions.map((instruction, i) => {
    const detail = instruction.op === 'argument' ? `argument ${instruction.index}`
      : instruction.op === 'external' ? instruction.name
      : instruction.op === 'literal' || instruction.op === 'bigint' ? JSON.stringify(instruction.value)
      : instruction.op === 'binary' ? instruction.operator
      : instruction.op === 'call' ? `target i${instruction.target}`
      : instruction.op === 'member' ? `object i${instruction.object} · key i${instruction.key}` : '';
    return { id: `i${i}`, title: instruction.op === 'argument' ? `arg ${instruction.index}` : instruction.op === 'external' ? instruction.name : instruction.op === 'binary' ? instruction.operator : instruction.op, detail, inputs: refs(instruction), input: instruction.op === 'argument', output: i === ir.result };
  });
}

/** Full value graph; arrows point from input values to consuming instructions. */
export function IRGraph(props: { name: string; nodes: GraphNode[]; selected?: string; onSelect: (id: string) => void }) {
  let viewport!: HTMLDivElement;
  const [zoom, setZoom] = createSignal(1);
  const [fontVersion, setFontVersion] = createSignal(0);
  const measure = document.createElement('canvas').getContext('2d');
  const textWidth = (text: string, size: number) => {
    if (!measure) return text.length * size * 0.65;
    measure.font = `${size}px 'IBM Plex Mono', monospace`;
    return measure.measureText(text).width;
  };
  const arrow = `ir-arrow-${createUniqueId().replace(/[^\w-]/g, '')}`;
  const layout = createMemo(() => {
    fontVersion();
    const graph = new dagre.graphlib.Graph({ multigraph: true });
    graph.setGraph({ rankdir: 'TB', nodesep: 12, ranksep: 28, marginx: 12, marginy: 12 });
    graph.setDefaultEdgeLabel(() => ({}));
    const ids = new Set(props.nodes.map(n => n.id));
    for (const node of props.nodes) graph.setNode(node.id, {
      width: Math.ceil(textWidth(node.id.replace(/^buffer:/, ''), 11) + textWidth(node.title, 12) + 30),
      height: 36,
    });
    for (const node of props.nodes) node.inputs.forEach((input, i) => {
      if (ids.has(input.id)) graph.setEdge(input.id, node.id, { label: input.label }, String(i));
    });
    dagre.layout(graph);
    return {
      width: Math.max(240, graph.graph().width || 0), height: Math.max(160, graph.graph().height || 0),
      nodes: props.nodes.map(node => ({ ...node, ...graph.node(node.id) as { x: number; y: number; width: number; height: number } })),
      edges: graph.edges().map(edge => ({ from: edge.v, to: edge.w, ...graph.edge(edge) as { label: string; points: { x: number; y: number }[] } })),
    };
  });
  const fit = () => {
    const graph = layout();
    setZoom(Math.min(1, Math.max(0.02, Math.min((viewport.clientWidth - 8) / graph.width, (viewport.clientHeight - 8) / graph.height))));
    viewport.scrollTo(0, 0);
  };
  createEffect(() => {
    layout();
    const frame = requestAnimationFrame(fit);
    onCleanup(() => cancelAnimationFrame(frame));
  });
  onMount(() => {
    let disposed = false;
    void document.fonts.ready.then(() => { if (!disposed) setFontVersion(v => v + 1); });
    let width = viewport.offsetWidth, height = viewport.offsetHeight;
    const observer = new ResizeObserver(() => {
      const nextWidth = viewport.offsetWidth, nextHeight = viewport.offsetHeight;
      if (nextWidth === width && nextHeight === height) return;
      width = nextWidth; height = nextHeight;
      fit();
    });
    observer.observe(viewport, { box: 'border-box' });
    onCleanup(() => { disposed = true; observer.disconnect(); });
  });
  createEffect(() => {
    const node = layout().nodes.find(n => n.id === props.selected);
    const scale = zoom();
    if (!node) return;
    const frame = requestAnimationFrame(() => viewport.scrollTo({
      left: Math.max(0, node.x * scale - viewport.clientWidth / 2),
      top: Math.max(0, node.y * scale - viewport.clientHeight / 2),
    }));
    onCleanup(() => cancelAnimationFrame(frame));
  });
  const related = (id: string) => id === props.selected || props.nodes.some(n => n.id === props.selected && n.inputs.some(input => input.id === id)) || props.nodes.some(n => n.id === id && n.inputs.some(input => input.id === props.selected));
  return <div class="ir-graph">
    <div class="graph-toolbar"><span class="hint">{props.nodes.length} nodes · {layout().edges.length} edges</span><div><button aria-label="Zoom out" onClick={() => setZoom(z => Math.max(0.02, z - 0.1))}>−</button><span class="hint">{Math.round(zoom() * 100)}%</span><button aria-label="Zoom in" onClick={() => setZoom(z => Math.min(2, z + 0.1))}>+</button></div></div>
    <div class="graph-viewport" ref={viewport}>
      <svg class="ir-svg" width={layout().width * zoom()} height={layout().height * zoom()} viewBox={`0 0 ${layout().width} ${layout().height}`} role="group" aria-label={`${props.name} full graph`}>
        <defs><marker id={arrow} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="context-stroke" /></marker></defs>
        <For each={layout().edges}>{edge => {
          const active = () => edge.from === props.selected || edge.to === props.selected;
          const mid = edge.points[Math.floor(edge.points.length / 2)]!;
          return <g class="ir-edge" classList={{ active: active() }} data-from={edge.from} data-to={edge.to}>
            <path d={edge.points.map((p, i) => `${i ? 'L' : 'M'}${p.x},${p.y}`).join(' ')} marker-end={`url(#${arrow})`} />
            <text x={mid.x + 6} y={mid.y - 6} classList={{ visible: active() }}>{edge.label}</text>
          </g>;
        }}</For>
        <For each={layout().nodes}>{node => <foreignObject x={node.x - node.width / 2} y={node.y - node.height / 2} width={node.width} height={node.height}>
          <button class="ir-node" classList={{ selected: props.selected === node.id, related: related(node.id), output: node.output, input: node.input }} data-node={node.id} aria-pressed={props.selected === node.id} title={`${node.id} · ${node.title} · ${node.detail}`} aria-label={`${node.id} ${node.title} ${node.detail}`} onClick={() => props.onSelect(node.id)}>
            <span class="ir-node-id">{node.id.replace(/^buffer:/, '')}</span><strong>{node.title}</strong>
          </button>
        </foreignObject>}</For>
      </svg>
    </div>
  </div>;
}
