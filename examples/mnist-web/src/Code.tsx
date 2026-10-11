import { createMemo, For, Show } from 'solid-js';
import Prism from 'prismjs';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-json';
import type { KernelSource } from './trace-model';

Prism.manual = true;
Prism.languages.wgsl = Prism.languages.extend('clike', {
  keyword: /\b(?:alias|break|case|const|const_assert|continue|continuing|default|diagnostic|discard|else|enable|false|fn|for|if|let|loop|override|requires|return|struct|switch|true|var|while)\b/,
  'class-name': /\b(?:array|atomic|bool|f16|f32|i32|u32|vec[234][fiuhb]?|mat[234]x[234][fh]?|ptr|sampler|texture_\w+)\b/,
  annotation: /@\w+/,
});

interface Token { text: string; classes: string; start: number; end: number }
function tokenize(source: string, language: string): Token[] {
  const tokens: Token[] = []; let offset = 0;
  const visit = (value: string | Prism.Token | (string | Prism.Token)[], classes = '') => {
    if (Array.isArray(value)) { value.forEach(v => visit(v, classes)); return; }
    if (typeof value !== 'string') { visit(value.content, `${classes} token ${value.type} ${Array.isArray(value.alias) ? value.alias.join(' ') : value.alias || ''}`); return; }
    tokens.push({ text: value, classes, start: offset, end: offset + value.length }); offset += value.length;
  };
  visit(Prism.tokenize(source, Prism.languages[language] || {}));
  return tokens;
}

/** Text tokens keep source offsets and render safely without innerHTML. */
export function Code(props: { source: string; language: string; kernels?: KernelSource[]; selected?: KernelSource; onSelect?: (kernel: KernelSource, event: MouseEvent) => void; highlight?: { start: number; end: number }; highlights?: readonly { start: number; end: number }[] }) {
  const tokens = createMemo(() => tokenize(props.source, props.language));
  const segments = createMemo(() => {
    const parts: { start: number; end: number; kernel?: KernelSource }[] = []; let start = 0;
    for (const kernel of (props.kernels || []).filter(k => k.start >= 0 && k.end <= props.source.length).sort((a, b) => a.start - b.start)) {
      if (kernel.start < start) continue;
      const lineStart = props.source.lastIndexOf('\n', kernel.start - 1) + 1;
      const clickStart = lineStart >= start && /^\s*(?:export(?:\s+default)?\s+)?$/.test(props.source.slice(lineStart, kernel.start)) ? lineStart : kernel.start;
      if (clickStart > start) parts.push({ start, end: clickStart });
      parts.push({ start: clickStart, end: kernel.end, kernel }); start = kernel.end;
    }
    if (start < props.source.length) parts.push({ start, end: props.source.length });
    return parts;
  });
  const text = (start: number, end: number) => {
    const pieces = createMemo(() => tokens().filter(t => t.start < end && t.end > start).flatMap(token => {
      const a = Math.max(start, token.start), b = Math.min(end, token.end);
      const ranges = props.highlights || (props.highlight ? [props.highlight] : []);
      const boundaries = [...new Set([a, b, ...ranges.flatMap(range => [range.start, range.end].filter(n => n > a && n < b))])].sort((x, y) => x - y);
      return boundaries.slice(0, -1).map((from, i) => ({
        classes: token.classes + (ranges.some(range => from >= range.start && from < range.end) ? ' highlighted' : ''),
        text: token.text.slice(from - token.start, boundaries[i + 1]! - token.start),
      }));
    }));
    return <For each={pieces()}>{piece => <span class={piece.classes}>{piece.text}</span>}</For>;
  };
  return <code class={`syntax language-${props.language}`}><For each={segments()}>{part => <Show when={part.kernel} fallback={text(part.start, part.end)}>{kernel => <button type="button" class="kernel-code" aria-label={`${kernel().name} compilation results`} aria-pressed={props.selected?.file === kernel().file && props.selected?.start === kernel().start} onClick={event => props.onSelect?.(kernel(), event)}>{text(part.start, part.end)}</button>}</Show>}</For></code>;
}
