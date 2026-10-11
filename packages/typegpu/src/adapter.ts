import tgpu, { type TgpuComputePipeline, type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import type { Block, Expression, Statement } from 'tinyest';
import type { DType } from 'blaten';

// The only private TypeGPU integration point. Supports TypeGPU ~0.12.7 / tinyest 0.3.4.
// Application IR stays on blaten's kernel Symbol; only generated compute AST goes here.
export function pipeline(root: TgpuRoot, name: string, body: Statement[], resources: Record<string, unknown>, onCode?: (code: string) => void): TgpuComputePipeline {
  const stub = (_input: { id: d.v3u }) => { throw new Error('Generated kernels execute only on the GPU') };
  const globals = globalThis as typeof globalThis & { __TYPEGPU_META__?: WeakMap<Function, unknown> };
  const externals = Object.fromEntries(Object.entries({ d, std, ...resources }).map(([key, value]) => [key, () => value]));
  (globals.__TYPEGPU_META__ ??= new WeakMap()).set(stub, {
    v: 2, name, ast: { params: [{ type: 'i', name: 'input' }], body: [0, body] satisfies Block }, externals,
  });
  const pipeline = root.createComputePipeline({ compute: tgpu.computeFn({ in: { id: d.builtin.globalInvocationId }, workgroupSize: [64] })(stub).$name(name) });
  if (onCode) {
    // TypeGPU 0.12.7 keeps the exact resolved WGSL in this result, but exposes
    // no public code-generation callback. Observe this pipeline's resolution
    // inside the pinned adapter; never patch GPUDevice or resolve a second time.
    const key = Object.getOwnPropertySymbols(pipeline).find(symbol => symbol.description === 'typegpu:0.12.7:$internal');
    if (!key) throw new Error('Unsupported TypeGPU trace integration: expected 0.12.7 pipeline internals');
    const core = (pipeline as unknown as Record<symbol, { core: unknown }>)[key]!.core as {
      resolveAndCreateShaderModule(): { resolutionResult: { code: string } };
    };
    const resolve = core.resolveAndCreateShaderModule;
    core.resolveAndCreateShaderModule = function () {
      const result = resolve.call(this);
      onCode(result.resolutionResult.code);
      return result;
    };
  }
  return pipeline;
}
export type E = Expression;
export type S = Statement;
export const num = (n: number): E => [5, String(n)];
export const member = (x: E, key: string): E => [7, x, key];
export const call = (x: E, ...args: E[]): E => [6, x, args];
export const uint = (n: number): E => call(member('d', 'u32'), num(n));
export const float = (n: number): E => call(member('d', 'f32'), num(n));
export function scalar(dtype: DType, value: E, source?: DType): E {
  if (dtype === 'bool') return binary(value, '!=', uint(0));
  const type = dtype.startsWith('i') ? 'i32' : dtype.startsWith('u') ? 'u32' : dtype;
  if ((dtype.startsWith('i') || dtype.startsWith('u')) && (source === 'f16' || source === 'f32')) {
    // TypedArray-style integer casts wrap instead of WGSL's out-of-range saturation.
    const floating = call(member('d', 'f32'), value);
    const magnitude = call(member('d', 'u32'), binary(call(member('std', 'abs'), floating), '%', float(4294967296)));
    value = call(member('std', 'select'), magnitude, binary(uint(0), '-', magnitude), binary(floating, '<', float(0)));
  }
  let result = call(member('d', type), value);
  if (dtype === 'i8' || dtype === 'i16') {
    const shift = uint(32 - Number(dtype.slice(1)));
    result = binary(binary(result, '<<', shift), '>>', shift);
  } else if (dtype === 'u8' || dtype === 'u16') result = binary(result, '&', uint(2 ** Number(dtype.slice(1)) - 1));
  return result;
}
export const binary = (a: E, op: any, b: E): E => [1, a, op === '!=' ? '!==' : op, b];
export const intdiv = (a: E, b: E): E => call(member('std', 'intdiv'), a, b);
export const at = (name: string, index: E): E => [8, member(name, '$'), index];
export const assign = (a: E, value: E): S => [2, a, '=', value];
export const block = (body: S[]): Block => [0, body];
export const loop = (name: string, count: number, body: S[]): S => [14, [12, name, uint(0)], binary(name, '<', uint(count)), [102, '++', name], block(body)];
export const index = member(member('input', 'id'), 'x');
