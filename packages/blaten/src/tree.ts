import { Tensor } from './tensor.js';

export type TensorTree = Tensor | readonly TensorTree[] | { readonly [key: string]: TensorTree };
export type TreeLayout = { kind: 'leaf' } | { kind: 'tuple'; items: TreeLayout[] } | { kind: 'record'; fields: Record<string, TreeLayout> };

/** Fixed arrays and plain records only; tensor identity and object key order are not layouts. */
export function flattenTree<T>(value: unknown, leaf: (value: unknown) => value is T): { leaves: T[]; layout: TreeLayout } {
  const leaves: T[] = [], active = new Set<object>();
  const visit = (value: unknown): TreeLayout => {
    if (leaf(value)) { leaves.push(value); return { kind: 'leaf' } }
    if (!value || typeof value !== 'object' || (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)) throw new Error('Expected a tensor/scalar, fixed array, or plain record');
    if (active.has(value)) throw new Error('Tensor trees cannot contain cycles');
    active.add(value);
    let result: TreeLayout;
    if (Array.isArray(value)) {
      if (Object.keys(value).length !== value.length) throw new Error('Tensor trees cannot contain sparse arrays');
      result = { kind: 'tuple', items: value.map(visit) };
    } else {
      const fields: Record<string, TreeLayout> = {};
      for (const key of Object.keys(value).sort()) {
        if (key === '__proto__') throw new Error('Unsupported tensor tree key');
        fields[key] = visit((value as Record<string, unknown>)[key]);
      }
      result = { kind: 'record', fields };
    }
    active.delete(value); return result;
  };
  return { leaves, layout: visit(value) };
}
export function unflattenTree<T>(layout: TreeLayout, leaves: readonly T[]): any {
  let index = 0;
  const visit = (layout: TreeLayout): any => layout.kind === 'leaf' ? leaves[index++] : layout.kind === 'tuple' ? layout.items.map(visit) : Object.fromEntries(Object.entries(layout.fields).map(([key, child]) => [key, visit(child)]));
  const result = visit(layout);
  if (index !== leaves.length) throw new Error('Tensor tree leaf count does not match');
  return result;
}
export function tensors(tree: unknown): { leaves: Tensor[]; layout: TreeLayout } { return flattenTree(tree, (value): value is Tensor => value instanceof Tensor) }
export function sameLayout(a: TreeLayout, b: TreeLayout): boolean { return JSON.stringify(a) === JSON.stringify(b) }
export function disposeTree(tree: TensorTree): void { new Set(tensors(tree).leaves).forEach(value => value.dispose()) }
