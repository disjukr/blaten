import { Tensor } from './tensor.js';

interface Frame { tensors: Set<Tensor>; parent?: Frame }
let current: Frame | undefined;
export function trackTensor(tensor: Tensor): void { current?.tensors.add(tensor) }
/** Transfer persistent storage (e.g. accumulator buffers) out of the current scope. */
export function keepTensor(tensor: Tensor): void { current?.tensors.delete(tensor) }
function returnedTensors(value: unknown, found = new Set<Tensor>(), seen = new Set<object>()): Set<Tensor> {
  if (!value || typeof value !== 'object' || seen.has(value)) return found;
  seen.add(value);
  if (value instanceof Tensor) found.add(value);
  else if (Array.isArray(value)) value.forEach(child => returnedTensors(child, found, seen));
  else if (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null) Object.values(value).forEach(child => returnedTensors(child, found, seen));
  return found;
}
/** Allocations are synchronous; a returned read Promise settles before cleanup. */
export function scope<T>(fn: () => T): T {
  if (fn.constructor.name === 'AsyncFunction') throw new Error('scope requires a synchronous callback; return a read Promise instead of awaiting inside the callback');
  const frame: Frame = { tensors: new Set(), parent: current };
  const finish = (value?: unknown) => {
    const kept = returnedTensors(value);
    for (const tensor of frame.tensors) {
      if (kept.has(tensor)) frame.parent?.tensors.add(tensor);
      else tensor.dispose();
    }
    frame.tensors.clear();
  };
  current = frame;
  let result: T;
  try { result = fn() }
  catch (error) { finish(); throw error }
  finally { current = frame.parent }
  if (result instanceof Promise) return result.then(value => { finish(value); return value }, error => { finish(); throw error }) as T;
  finish(result); return result;
}
