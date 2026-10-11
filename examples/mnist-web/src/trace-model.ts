import type { Compilation, TraceResult } from 'blaten/trace';

export interface KernelSource { name: string; file: string; start: number; end: number }
export interface Kernel {
  id: string; kind: string; name?: string; value?: boolean;
  source?: string | { file: string; start: number; end: number; artifact?: string };
  externals?: Record<string, { kernel?: string }>;
  backward?: string; jvp?: string;
}
export interface KernelGraph { entry: string; kernels: Kernel[] }
export interface KernelStep { id: string; name: string; relation?: string }

export function kernelGraph(compilation: Compilation): KernelGraph | undefined {
  const content = compilation.artifacts.find(a => a.kind === 'blaten.kernel-ir')?.content;
  return content?.type === 'json' ? content.value as unknown as KernelGraph : undefined;
}

export function kernelSources(result?: TraceResult): KernelSource[] {
  const sources = new Map<string, KernelSource>();
  for (const c of result?.compilations || []) for (const k of kernelGraph(c)?.kernels || []) {
    if (k.kind !== 'kernel' || !k.source || typeof k.source === 'string') continue;
    const { file, start, end } = k.source;
    sources.set(`${file}:${start}:${end}`, { file, start, end, name: k.name || k.id });
  }
  return [...sources.values()].sort((a, b) => a.start - b.start);
}

/** Follow recorded call/derivative references, never infer per-node source provenance. */
export function kernelPaths(compilation: Compilation, source: KernelSource): KernelStep[][] {
  const graph = kernelGraph(compilation); if (!graph) return [];
  const ids = new Map(graph.kernels.map(k => [k.id, k]));
  const root = graph.kernels.find(k => k.kind === 'kernel' && typeof k.source === 'object' && k.source.file === source.file && k.source.start === source.start && k.source.end === source.end);
  if (!root) return [];
  const nameOf = (id: string, seen = new Set<string>()): string => {
    const k = ids.get(id); if (!k || seen.has(id)) return id;
    if (k.kind === 'kernel') return k.name || id;
    seen.add(id);
    const name = typeof k.source === 'string' ? nameOf(k.source, seen) : id;
    return `${k.kind === 'grad' ? k.value ? 'valueAndGrad' : 'grad' : k.kind}(${name})`;
  };
  const paths: KernelStep[][] = [];
  const descend = (id: string, path: KernelStep[]) => {
    const k = ids.get(id)!;
    const references = [
      ...(typeof k.source === 'string' ? [{ id: k.source, relation: k.kind === 'grad' ? 'Differentiated function' : k.kind }] : []),
      ...Object.values(k.externals || {}).flatMap(ref => ref.kernel ? [{ id: ref.kernel, relation: 'Call' }] : []),
      ...(k.backward ? [{ id: k.backward, relation: 'backward' }] : []),
      ...(k.jvp ? [{ id: k.jvp, relation: 'jvp' }] : []),
    ].filter(ref => ids.has(ref.id) && !path.some(p => p.id === ref.id));
    if (!references.length) { paths.push(path); return; }
    for (const ref of references) descend(ref.id, [...path, { ...ref, name: nameOf(ref.id) }]);
  };
  const visit = (id: string, path: KernelStep[]) => {
    if (id === graph.entry) {
      descend(root.id, [...path].reverse().map((step, i, reversed) => ({
        id: step.id, name: step.name,
        ...(i ? { relation: reversed[i - 1]!.relation === 'Autodiff' ? 'Differentiated function' : reversed[i - 1]!.relation } : {}),
      })));
      return;
    }
    for (const k of graph.kernels) {
      if (path.some(p => p.id === k.id)) continue;
      const relation = k.source === id ? k.kind === 'grad' ? 'Autodiff' : k.kind
        : Object.values(k.externals || {}).some(ref => ref.kernel === id) ? 'Call'
        : k.backward === id ? 'backward' : k.jvp === id ? 'jvp' : undefined;
      if (relation) visit(k.id, [...path, { id: k.id, name: nameOf(k.id), relation }]);
    }
  };
  visit(root.id, [{ id: root.id, name: nameOf(root.id) }]);
  return paths;
}
