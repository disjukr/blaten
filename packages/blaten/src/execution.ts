import { Tensor, storageOf, hostData } from './tensor.js';
import { isHostStorage, type Scalar, type HostStorage } from './types.js';
import type { Program } from './lowering.js';
import { evaluateCpu } from './intrinsics.js';
import type { TensorOrScalar } from './runtime.js';
import { makeStorage } from './types.js';

interface Environment {
  execute(program: Program, inputs: readonly TensorOrScalar[]): Tensor | Tensor[];
  assign(target: Tensor, source: Tensor): void;
}
const environments = new WeakMap<object, Environment>();
export function registerEnvironment(owner: object, environment: Environment): void { environments.set(owner, environment) }

export function execute(program: Program, inputs: readonly TensorOrScalar[]): Tensor | Tensor[] {
  const owners = new Set(inputs.flatMap(input => {
    if (!(input instanceof Tensor)) return [];
    const storage = storageOf(input);
    return isHostStorage(storage) ? [] : [storage.owner];
  }));
  if (owners.size > 1) throw new Error('Tensor operations cannot mix backend environments; use copyFrom() first');
  if (owners.size) {
    if (inputs.some(input => input instanceof Tensor && isHostStorage(storageOf(input)))) throw new Error('Tensor operations cannot mix host and backend tensors; use copyFrom() first');
    const environment = environments.get([...owners][0]!);
    if (!environment) throw new Error('This tensor environment does not support direct operations');
    return environment.execute(program, inputs);
  }
  const bindings = new Map(program.inputs.map((node, index) => {
    const input = inputs[index]!;
    const values: HostStorage = input instanceof Tensor ? hostData(input) : makeStorage(node.dtype, [input as Scalar]);
    return [node, values] as const;
  }));
  const results = program.outputs.map(node => new Tensor(evaluateCpu(node, bindings), node.shape, node.dtype));
  return program.single ? results[0]! : results;
}

export function assign(target: Tensor, source: Tensor): void {
  const storage = storageOf(target), sourceStorage = storageOf(source);
  if (target.dtype !== source.dtype || String(target.shape) !== String(source.shape)) throw new Error('Assignment requires matching shape and dtype');
  if (isHostStorage(storage)) {
    if (!isHostStorage(sourceStorage)) throw new Error('Assignment cannot implicitly read backend data');
    (storage as any).set(sourceStorage);
  } else {
    if (isHostStorage(sourceStorage) || storage.owner !== sourceStorage.owner) throw new Error('Assignment requires the same environment');
    const environment = environments.get(storage.owner);
    if (!environment) throw new Error('This tensor environment does not support assignment');
    environment.assign(target, source);
  }
}
