import tgpu, { type TgpuRoot, type TgpuComputePipeline } from 'typegpu';
import * as d from 'typegpu/data';
import { Tensor, type TensorTree, type Shape, type DType, type HostStorage, type Scalar } from 'blaten';
import { lower, signatureOf, topological, sizeOf, isHostStorage, isFloat, makeStorage, type Node, type KernelFunction, type Backend, type Executable, type Program, type CompileOptions, type TensorOrScalar, storageOf, hostData, deviceStorageOf, registerEnvironment, callable, isKernelLeaf, flattenTree, unflattenTree, sameLayout, scope, tensors, Resource } from 'blaten/backend';
import * as ast from './adapter.js';
const diagnostics = new WeakMap<Executable, { readonly dispatchCount: number; readonly workspaceBytes: number }>();
export function inspect(executable: Executable) {
  const result = diagnostics.get(executable);
  if (!result) throw new Error('Executable was not created by the TypeGPU backend');
  return result;
}
type Buffer = ReturnType<TgpuRoot['createBuffer']>;
const sameShape = (a: Shape, b: Shape) => String(a) === String(b);
const storageType = (dtype: DType) => dtype === 'bool' || dtype.startsWith('u') ? 'u32' : dtype.startsWith('i') ? 'i32' : dtype;
const gpuBytes = (count: number, dtype: DType) => dtype === 'f16' ? Math.ceil(count / 2) * 4 : count * 4;
// Bit tests preserve NaN handling on devices that optimize x != x away.
function isNaNValue(value: ast.E): ast.E {
  const bits = ast.call(ast.call(ast.member('std', 'bitcast'), ast.member('d', 'f32'), ast.member('d', 'u32')), ast.scalar('f32', value));
  return ast.binary(ast.binary(bits, '&', ast.uint(0x7fffffff)), '>', ast.uint(0x7f800000));
}
function coordinate(index: ast.E, shape: Shape, axis: number): ast.E {
  const stride = sizeOf(shape.slice(axis + 1));
  return ast.binary(stride === 1 ? index : ast.intdiv(index, ast.uint(stride)), '%', ast.uint(shape[axis]!));
}
function broadcastIndex(index: ast.E, out: Shape, input: Shape): ast.E {
  if (sameShape(out, input)) return index;
  if (!input.length || sizeOf(input) === 1) return ast.uint(0);
  let result = ast.uint(0);
  input.forEach((dim, i) => {
    if (dim === 1) return;
    const part = coordinate(index, out, i + out.length - input.length);
    const stride = sizeOf(input.slice(i + 1));
    result = ast.binary(result, '+', stride === 1 ? part : ast.binary(part, '*', ast.uint(stride)));
  });
  return result;
}

export class TypeGpuBackend implements Backend {
  readonly name = 'typegpu';
  private destroyed = false;
  private readonly resources = new Set<Resource>();

  private readonly workspace = new Map<string, Buffer[]>();
  private cachedBytes = 0;
  private readonly directPlans = new Map<string, Executable>();
  constructor(readonly root: TgpuRoot, private readonly ownsDevice = false) {
    registerEnvironment(this, {
      execute: (program, inputs) => this.executeDirect(program, inputs),
      assign: (target, source) => {
        this.assertAlive();
        const encoder = this.root.device.createCommandEncoder();
        encoder.copyBufferToBuffer((deviceStorageOf(source).buffer as Buffer).buffer, 0, (deviceStorageOf(target).buffer as Buffer).buffer, 0, gpuBytes(source.size, source.dtype));
        this.root.device.queue.submit([encoder.finish()]);
      },
    });
  }
  private assertAlive() { if (this.destroyed) throw new Error('Backend is disposed') }
  get capabilities(): Backend['capabilities'] {
    return { dtypes: ['bool', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'f32', ...(this.root.device.features.has('shader-f16') ? ['f16' as const] : [])] };
  }
  private assertDType(dtype: DType) {
    if (!this.capabilities.dtypes.includes(dtype)) throw new Error(`TypeGPU backend does not support ${dtype}${dtype === 'f16' ? ': device requires shader-f16' : ''}`);
  }
  private write(buffer: Buffer, values: HostStorage, dtype: DType): void {
    const packed = storageType(dtype) === dtype ? values : makeStorage(storageType(dtype) as DType, Array.from(values as any) as Scalar[]);
    const bytes = new Uint8Array(gpuBytes(values.length, dtype));
    bytes.set(new Uint8Array(packed.buffer, packed.byteOffset, packed.byteLength));
    buffer.write(bytes.buffer);
  }
  private buffer(shape: Shape, dtype: DType, values?: HostStorage): Buffer {
    this.assertDType(dtype);
    const bytes = gpuBytes(sizeOf(shape), dtype);
    if (bytes > Math.min(this.root.device.limits.maxBufferSize, this.root.device.limits.maxStorageBufferBindingSize)) throw new Error('Tensor exceeds device storage-buffer limits');
    const type = storageType(dtype) as 'f16' | 'f32' | 'i32' | 'u32';
    const buffer = this.root.createBuffer(d.arrayOf(d[type], dtype === 'f16' ? Math.ceil(sizeOf(shape) / 2) * 2 : sizeOf(shape))).$usage('storage');
    if (values) this.write(buffer as Buffer, values, dtype);
    return buffer as Buffer;
  }
  private trackResource(cleanup: () => void): Resource {
    const resource = new Resource(() => {
      try { cleanup() } finally { this.resources.delete(resource) }
    });
    this.resources.add(resource); return resource;
  }
  private deviceTensor<S extends Shape, D extends DType>(buffer: Buffer, shape: S, dtype: D): Tensor<S, D> {
    const resource = this.trackResource(() => buffer.destroy());
    return new Tensor({ owner: this, buffer,
      get isDisposed() { return resource.isDisposed },
      read: async () => { this.assertAlive(); resource.assertAlive(); return makeStorage(dtype, (await buffer.read() as number[]).slice(0, sizeOf(shape))) },
      dispose: () => resource.dispose(),
    }, shape, dtype);
  }
  scope = scope;
  copyFrom<S extends Shape, D extends DType>(value: Tensor<S, D>): Promise<Tensor<S, D>>;
  copyFrom<T extends TensorTree>(value: T): Promise<T>;
  async copyFrom(value: TensorTree): Promise<any> {
    this.assertAlive();
    if (!(value instanceof Tensor)) {
      const { leaves, layout } = tensors(value);
      const results = await Promise.allSettled(leaves.map(leaf => this.copyFrom(leaf)));
      const failure = results.find(result => result.status === 'rejected');
      if (failure?.status === 'rejected') {
        results.forEach(result => { if (result.status === 'fulfilled') result.value.dispose() });
        throw failure.reason;
      }
      return unflattenTree(layout, results.map(result => (result as PromiseFulfilledResult<Tensor>).value));
    }
    this.assertAlive(); value.assertAlive();
    this.assertDType(value.dtype);
    if (isHostStorage(storageOf(value))) {
      return this.deviceTensor(this.buffer(value.shape, value.dtype, hostData(value)), value.shape, value.dtype);
    }
    if (deviceStorageOf(value).owner === this) {
      const buffer = this.buffer(value.shape, value.dtype);
      try {
        const encoder = this.root.device.createCommandEncoder();
        encoder.copyBufferToBuffer((deviceStorageOf(value).buffer as Buffer).buffer, 0, buffer.buffer, 0, gpuBytes(value.size, value.dtype));
        this.root.device.queue.submit([encoder.finish()]);
        return this.deviceTensor(buffer, value.shape, value.dtype);
      } catch (error) { buffer.destroy(); throw error }
    }
    // Other environments provide a host snapshot; no knowledge of their buffer
    // representation is required. Recheck lifetime after the asynchronous read.
    const data = await value.read();
    this.assertAlive(); value.assertAlive();
    return this.deviceTensor(this.buffer(value.shape, value.dtype, data), value.shape, value.dtype);
  }
  async compile<T extends KernelFunction>(fn: T, options?: CompileOptions): Promise<Executable<T>> {
    this.assertAlive();
    return this.compileProgram(lower(fn, signatureOf(fn, options?.inputs))) as Executable<T>;
  }
  async synchronize(): Promise<void> {
    this.assertAlive(); await this.root.device.queue.onSubmittedWorkDone();
  }
  private acquireWorkspace(shape: Shape, dtype: DType): Buffer {
    const key = dtype + ':' + sizeOf(shape), free = this.workspace.get(key), buffer = free?.pop();
    if (buffer) { if (!free!.length) this.workspace.delete(key); this.cachedBytes -= gpuBytes(sizeOf(shape), dtype); return buffer }
    return this.buffer(shape, dtype);
  }
  private releaseWorkspace(buffer: Buffer, shape: Shape, dtype: DType): void {
    const bytes = gpuBytes(sizeOf(shape), dtype);
    if (this.destroyed || this.cachedBytes + bytes > 64 * 1024 * 1024) { buffer.destroy(); return }
    const key = dtype + ':' + sizeOf(shape), free = this.workspace.get(key) ?? [];
    free.push(buffer); this.workspace.set(key, free); this.cachedBytes += bytes;
  }
  private executeDirect(program: Program, inputs: readonly TensorOrScalar[]): Tensor | Tensor[] {
    this.assertAlive();
    const nodes = topological(program.outputs), indexes = new Map(nodes.map((node, index) => [node, index]));
    const key = JSON.stringify([program.single, program.inputs.map(node => indexes.get(node)), nodes.map(node => [node.op, node.shape, node.dtype, node.inputs.map(input => indexes.get(input)), Object.fromEntries(Object.entries(node.attrs).filter(([name]) => name !== 'tensor'))])], (_, value) => typeof value === 'bigint' ? 'bigint:' + value : typeof value === 'number' && !Number.isFinite(value) ? String(value) : value);
    // Plans bind inputs on each run; do not retain the first caller's tensors.
    nodes.forEach(node => { delete node.attrs.tensor });
    let executable = this.directPlans.get(key);
    if (!executable) {
      executable = this.compileProgram(program);
      if (this.directPlans.size >= 32) {
        const oldest = this.directPlans.keys().next().value!;
        this.directPlans.get(oldest)!.dispose(); this.directPlans.delete(oldest);
      }
    }
    this.directPlans.delete(key); this.directPlans.set(key, executable);
    return executable(...inputs);
  }
  private compileProgram(program: Program): Executable {
    const order = topological(program.outputs);
    [...program.inputs, ...order].forEach(n => this.assertDType(n.dtype));
    const buffers = new Map<Node, Buffer>();
    const steps: { pipeline: TgpuComputePipeline; count: number }[] = [];
    const allocate = (n: Node, data?: HostStorage) => { const b = this.acquireWorkspace(n.shape, n.dtype); buffers.set(n, b); if (data) this.write(b, data, n.dtype); return b };
    // Materialize expensive operations and outputs; fuse elementwise/view IR into their consumers.
    const boundaries = new Set(['matmul', 'argmax', 'sum', 'softmax', 'logsoftmax']);
    try {
      program.inputs.forEach(n => allocate(n));
      for (const n of order) {
        if (n.op === 'constant' && n.attrs.data) { allocate(n, n.attrs.data); continue }
        if (n.op === 'constant' && isFloat(n.dtype)) {
          const values = makeStorage(n.dtype, [n.attrs.value]);
          // Nonfinite constants cannot be emitted as WGSL numeric literals.
          if (!Number.isFinite(Number(values[0]))) { allocate(n, values); continue }
        }
        if (buffers.has(n) || (!boundaries.has(n.op) && !program.outputs.includes(n))) continue;
        const output = allocate(n), resources: Record<string, unknown> = { output: (output as any).as('mutable') };
        const resourceNames = new Map<Node, string>();
        let sequence = 0;
        const emit = (node: Node, index: ast.E, statements: ast.S[], memo: Map<string, ast.E>): ast.E => {
          const key = `${node.id}:${JSON.stringify(index)}`;
          const cached = memo.get(key); if (cached) return cached;
          const existing = buffers.get(node);
          if (existing && node !== n) {
            let name = resourceNames.get(node);
            if (!name) { name = `buffer${resourceNames.size}`; resourceNames.set(node, name); resources[name] = (existing as any).as('readonly') }
            const value = ast.at(name, index);
            return node.dtype === 'bool' ? ast.binary(value, '!=', ast.uint(0)) : value;
          }
          const [a, b] = node.inputs;
          const read = (input: Node) => emit(input, broadcastIndex(index, node.shape, input.shape), statements, memo);
          let expression: ast.E;
          switch (node.op) {
            case 'constant': expression = node.dtype === 'bool' ? ast.binary(ast.uint(+Boolean(node.attrs.value)), '!=', ast.uint(0)) : ast.scalar(node.dtype, ast.num(Number(node.attrs.value))); break;
            case 'cast': expression = node.dtype === 'bool' ? ast.binary(read(a!), '!=', ast.scalar(a!.dtype, ast.num(0))) : ast.scalar(node.dtype, read(a!), a!.dtype); break;
            case 'custom': return emit(node.attrs.forward, index, statements, memo);
            case 'reshape': return emit(a!, index, statements, memo);
            case 'broadcast': return read(a!);
            case 'transpose': return emit(a!, ast.binary(ast.binary(ast.binary(index, '%', ast.uint(node.shape[1]!)), '*', ast.uint(a!.shape[1]!)), '+', ast.intdiv(index, ast.uint(node.shape[1]!))), statements, memo);
            case 'add': case 'sub': case 'mul': case 'div': expression = ast.binary(read(a!), { add: '+', sub: '-', mul: '*', div: '/' }[node.op], read(b!)); break;
            case 'eq': {
              const left = read(a!), right = read(b!);
              expression = ast.binary(left, '===', right);
              if (isFloat(a!.dtype)) {
                expression = ast.call(ast.member('std', 'select'), expression, false, isNaNValue(left));
                expression = ast.call(ast.member('std', 'select'), expression, false, isNaNValue(right));
              }
              break;
            }
            case 'gt': expression = ast.binary(read(a!), '>', read(b!)); break;
            case 'and': case 'or': expression = ast.binary(read(a!), node.op === 'and' ? '&&' : '||', read(b!)); break;
            case 'not': expression = [4, '!', read(a!)]; break;
            case 'neg': expression = [4, '-', read(a!)]; break;
            case 'exp': case 'log': case 'tanh': expression = ast.call(ast.member('std', node.op), read(a!)); break;
            case 'relu': expression = ast.call(ast.member('std', 'max'), read(a!), ast.scalar(node.dtype, ast.num(0))); break;
            default: throw new Error(`Cannot fuse operation ${node.op}`);
          }
          const name = `value${sequence++}`;
          if (node.dtype !== 'bool') expression = ast.scalar(node.dtype, expression);
          statements.push([13, name, expression]); memo.set(key, name); return name;
        };
        const body: ast.S[] = [[13, 'index', ast.index]];
        const work: ast.S[] = [];
        const [a, b] = n.inputs;
        const accumulatorDType = n.dtype === 'f16' ? 'f32' : n.dtype;
        let count = sizeOf(n.shape);
        if (n.op === 'matmul') {
          const columns = n.shape[1]!, inner = a!.shape[1]!;
          work.push([13, 'row', ast.intdiv('index', ast.uint(columns))], [13, 'column', ast.binary('index', '%', ast.uint(columns))], [12, 'total', ast.scalar(accumulatorDType, ast.num(0))]);
          const loopBody: ast.S[] = [], memo = new Map<string, ast.E>();
          const left = emit(a!, ast.binary(ast.binary('row', '*', ast.uint(inner)), '+', 'k'), loopBody, memo);
          const right = emit(b!, ast.binary(ast.binary('k', '*', ast.uint(columns)), '+', 'column'), loopBody, memo);
          loopBody.push([2, 'total', '+=', ast.binary(ast.scalar(accumulatorDType, left), '*', ast.scalar(accumulatorDType, right))]);
          work.push(ast.loop('k', inner, loopBody), ast.assign(ast.at('output', 'index'), ast.scalar(n.dtype, 'total')));
        } else if (n.op === 'argmax') {
          const axis: number = n.attrs.axis, width = a!.shape[axis]!;
          const stride = sizeOf(a!.shape.slice(axis + 1));
          const start = ast.binary(ast.binary(ast.intdiv('index', ast.uint(stride)), '*', ast.uint(width * stride)), '+', ast.binary('index', '%', ast.uint(stride)));
          work.push([13, 'start', start]);
          const initial = emit(a!, 'start', work, new Map());
          work.push([12, 'maximum', initial], [12, 'best', ast.uint(0)]);
          const loopBody: ast.S[] = [];
          const candidate = emit(a!, ast.binary('start', '+', ast.binary('k', '*', ast.uint(stride))), loopBody, new Map());
          loopBody.push([13, 'candidate', candidate]);
          loopBody.push([12, 'wins', ast.binary('candidate', '>', 'maximum')]);
          if (isFloat(a!.dtype)) {
            loopBody.push([13, 'maximumNaN', isNaNValue('maximum')], [13, 'candidateNaN', isNaNValue('candidate')]);
            loopBody.push([11, 'maximumNaN', ast.block([ast.assign('wins', false)])]);
            loopBody.push([11, 'candidateNaN', ast.block([
              [11, [4, '!', 'maximumNaN'], ast.block([ast.assign('wins', true)])],
            ])]);
          }
          loopBody.push([11, 'wins', ast.block([ast.assign('maximum', 'candidate'), ast.assign('best', 'k')])]);
          work.push(ast.loop('k', width, loopBody), ast.assign(ast.at('output', 'index'), ast.scalar('i32', 'best')));
        } else if (n.op === 'sum') {
          const axes: number[] = n.attrs.axes;
          const reduced = axes.map(i => a!.shape[i]!);
          const remaining = a!.shape.filter((_, i) => !axes.includes(i));
          work.push([12, 'total', ast.scalar(accumulatorDType, ast.num(0))]);
          const loopBody: ast.S[] = [], memo = new Map<string, ast.E>();
          let inputIndex = ast.uint(0), outputAxis = 0;
          a!.shape.forEach((dim, i) => {
            const axis = axes.indexOf(i);
            const part = axis >= 0 ? coordinate('k', reduced, axis) : coordinate('index', remaining, outputAxis++);
            inputIndex = ast.binary(ast.binary(inputIndex, '*', ast.uint(dim)), '+', part);
          });
          loopBody.push([2, 'total', '+=', ast.scalar(accumulatorDType, emit(a!, inputIndex, loopBody, memo))]);
          work.push(ast.loop('k', n.attrs.limit ?? sizeOf(reduced), loopBody), ast.assign(ast.at('output', 'index'), ast.scalar(n.dtype, 'total')));
        } else if (n.op === 'softmax' || n.op === 'logsoftmax') {
          const width = n.shape.at(-1)!; count /= width;
          work.push([13, 'start', ast.binary('index', '*', ast.uint(width))], [12, 'maximum', ast.scalar(accumulatorDType, ast.num(-3.4028234663852886e38))], [12, 'total', ast.scalar(accumulatorDType, ast.num(0))]);
          const position = ast.binary('start', '+', 'k');
          const maxBody: ast.S[] = [], expBody: ast.S[] = [], writeBody: ast.S[] = [];
          maxBody.push(ast.assign('maximum', ast.call(ast.member('std', 'max'), 'maximum', ast.scalar(accumulatorDType, emit(a!, position, maxBody, new Map())))));
          expBody.push([2, 'total', '+=', ast.call(ast.member('std', 'exp'), ast.binary(ast.scalar(accumulatorDType, emit(a!, position, expBody, new Map())), '-', 'maximum'))]);
          const x = ast.scalar(accumulatorDType, emit(a!, position, writeBody, new Map()));
          const value = n.op === 'softmax' ? ast.binary(ast.call(ast.member('std', 'exp'), ast.binary(x, '-', 'maximum')), '/', 'total') : ast.binary(ast.binary(x, '-', 'maximum'), '-', ast.call(ast.member('std', 'log'), 'total'));
          writeBody.push(ast.assign(ast.at('output', position), ast.scalar(n.dtype, value)));
          work.push(ast.loop('k', width, maxBody), ast.loop('k', width, expBody), ast.loop('k', width, writeBody));
        } else {
          const value = emit(n, 'index', work, new Map());
          work.push(ast.assign(ast.at('output', 'index'), n.dtype === 'bool' ? ast.call(ast.member('std', 'select'), ast.uint(0), ast.uint(1), value) : ast.scalar(n.dtype, value)));
        }
        body.push([11, ast.binary('index', '<', ast.uint(count)), ast.block(work)]);
        if (Math.ceil(count / 64) > this.root.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('Tensor exceeds device dispatch limits');
        const pipeline = ast.pipeline(this.root, `blaten_${n.op}_${n.id}`, body, resources);
        pipeline.initSync(); steps.push({ pipeline, count });
      }
    } catch (error) { for (const b of buffers.values()) b.destroy(); throw error }
    const resource = this.trackResource(() => {
      for (const [node, buffer] of buffers) this.releaseWorkspace(buffer, node.shape, node.dtype);
      buffers.clear(); steps.length = 0;
    });
    const executable: Executable = callable((...arguments_) => {
        let inputs = arguments_;
        if (program.inputLayout) {
          const tree = flattenTree(arguments_, isKernelLeaf);
          if (!sameLayout(tree.layout, program.inputLayout)) throw new Error('Input tree does not match signature');
          inputs = tree.leaves;
        }
        this.assertAlive(); resource.assertAlive();
        if (inputs.length !== program.inputs.length) throw new Error('Input count does not match signature');
        // Validate all inputs before recording or writing anything.
        const tensors = inputs.map((value, i) => {
          const expected = program.inputs[i]!.shape, dtype = program.inputs[i]!.dtype;
          if (typeof value !== 'object') {
            if (expected.length) throw new Error('Scalar supplied for tensor input');
            if ((dtype === 'bool') !== (typeof value === 'boolean') || typeof value === 'bigint') throw new Error('Scalar input does not match dtype');
            if (dtype.startsWith('i') || dtype.startsWith('u')) { if (!Number.isInteger(value)) throw new Error('Integer scalar input requires an integer') }
            return value;
          }
          value.assertAlive();
          if (!sameShape(value.shape, expected)) throw new Error(`Input ${i}: expected [${expected}], received [${value.shape}]`);
          if (value.dtype !== dtype) throw new Error(`Input ${i}: expected dtype ${dtype}, received ${value.dtype}`);
          if (!isHostStorage(storageOf(value)) && deviceStorageOf(value).owner !== this) throw new Error('Tensor belongs to a different backend');
          return value;
        });
        const encoder = this.root.device.createCommandEncoder();
        tensors.forEach((value, i) => {
          const buffer = buffers.get(program.inputs[i]!)!;
          const dtype = program.inputs[i]!.dtype;
          if (typeof value !== 'object') this.write(buffer, makeStorage(dtype, [value]), dtype);
          else if (isHostStorage(storageOf(value))) this.write(buffer, hostData(value), dtype);
          else encoder.copyBufferToBuffer((deviceStorageOf(value).buffer as Buffer).buffer, 0, buffer.buffer, 0, gpuBytes(value.size, dtype));
        });
        for (const step of steps) step.pipeline.with(encoder).dispatchWorkgroups(Math.ceil(step.count / 64));
        const results = program.outputs.map(n => {
          const snapshot = this.buffer(n.shape, n.dtype);
          encoder.copyBufferToBuffer(buffers.get(n)!.buffer, 0, snapshot.buffer, 0, gpuBytes(sizeOf(n.shape), n.dtype));
          return this.deviceTensor(snapshot, n.shape, n.dtype);
        });
        this.root.device.queue.submit([encoder.finish()]);
        return program.outputLayout ? unflattenTree(program.outputLayout, results) : program.single ? results[0]! : results;
      }, resource);
    diagnostics.set(executable, Object.freeze({ dispatchCount: steps.length, workspaceBytes: [...buffers.keys()].reduce((sum, node) => sum + gpuBytes(sizeOf(node.shape), node.dtype), 0) }));
    return executable;
  }
  dispose(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.resources.forEach(resource => resource.dispose());
    this.directPlans.clear();
    this.workspace.forEach(buffers => buffers.forEach(buffer => buffer.destroy())); this.workspace.clear(); this.cachedBytes = 0;
    this.root.destroy(); if (this.ownsDevice) this.root.device.destroy();
  }
}
export async function typegpu(options: { device?: GPUDevice } = {}): Promise<TypeGpuBackend> {
  if (options.device) return new TypeGpuBackend(tgpu.initFromDevice({ device: options.device }));
  if (!globalThis.navigator?.gpu) throw new Error('WebGPU is unavailable. In Node, provide a device from the webgpu package.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found');
  return new TypeGpuBackend(tgpu.initFromDevice({ device: await adapter.requestDevice({ requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [] }) }), true);
}
