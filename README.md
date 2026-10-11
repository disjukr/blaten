# blaten

An experimental TypeScript tensor library with kernel compilation and automatic differentiation. The first backend is TypeGPU.

```text
packages/
  blaten/             Tensors, IR, automatic differentiation, backend interfaces
  typegpu/            @blaten/typegpu: GPU code generation and execution
  compiler/           @blaten/compiler: AST-to-kernel-IR build plugin
  eslint-plugin/      @blaten/eslint-plugin: kernel syntax and helper validation
examples/
  mnist/              A 784 -> 32 -> 10 MLP trained on real MNIST data
```

Applications use `blaten` and `@blaten/typegpu`, with `@blaten/compiler` and `@blaten/eslint-plugin` as development dependencies. Babel is a dependency of both tooling packages; the compiler also uses the TypeScript type checker. The core runtime has no production dependencies on Babel, TypeScript, Vite, ESLint, TypeGPU, or tinyest. It consumes kernel IR, specializes shapes, constructs tensor IR, and performs automatic differentiation without retaining source ASTs.

Backends import lowering, IR types, and shared utilities from `blaten/backend`. The dependency direction is `@blaten/typegpu` -> `blaten`; the core does not import a particular backend.

TypeGPU is a peer dependency of `@blaten/typegpu`. Applications should depend on `typegpu: ~0.12.7` so they share one installation. This range allows patch releases before 0.13.0; backend development and testing currently use 0.12.7. The MNIST example records the installed TypeGPU version in its results.

## Getting started

Use Node.js 22.12 or later, pnpm 10, and a GPU/driver with WebGPU support. The Node examples use the `webgpu` package and Vite Module Runner. Host-side f16 tensors additionally require native `Float16Array` support.

```sh
pnpm install
pnpm build
pnpm test
pnpm typecheck
pnpm lint
pnpm mnist
```

MNIST downloads the [CVDF IDX files](https://storage.googleapis.com/cvdf-datasets/mnist/) on first use and caches them in `examples/mnist/data/`. Defaults are 12,000 training images, 2,000 separate test images, batch size 64, and three epochs. Use `TRAIN_SIZE=60000` and `TEST_SIZE=10000` for the full dataset. `EPOCHS` and `LEARNING_RATE` are also configurable through environment variables. `ACCUMULATION_STEPS` defaults to 1; set it to 4 to average four batches on the GPU before each update. The final accumulation window is flushed at the end of each epoch.

The model's `Images` and `Labels` types leave the batch dimension as `number`. Compilation specializes them from example inputs; set `BATCH_SIZE` to change the batch size without editing model wrappers. `model.ts` sets `hiddenSize = 32`. Results are written to `examples/mnist/results/latest.json`.

A verified run improved test accuracy from **12.65% to 88.75%** and epoch-average loss from **0.6756 to 0.2791**. The measurements and environment are recorded in [verification.json](examples/mnist/verification.json). Results depend on configuration and hardware.

## Usage

Add the build plugin to Vite. The example's [run.mjs](examples/mnist/run.mjs) connects the same plugin to Module Runner.

```ts
import { defineConfig } from 'vite';
import blaten from '@blaten/compiler/vite';

export default defineConfig({ plugins: [blaten()] });
```

```ts
import { tensor, f32, compile, grad, type Tensor, type Shape } from 'blaten';
import { typegpu, inspect } from '@blaten/typegpu';

function squaredSum(x: Tensor<[number], typeof f32>) {
  'use kernel';
  return x.mul(x).sum();
}

// Uses navigator.gpu in a browser.
const backend = await typegpu();
const input = tensor([1, 2, 3]);
const program = await compile(squaredSum, backend, { inputs: [input] });
const result = program(input);
console.log(await result.item()); // 14
console.log(inspect(program).dispatchCount); // 1: multiply and sum share a shader

const derivative = await compile(grad(squaredSum), backend, { inputs: [input] });
const gradient = derivative(input);
console.log(await gradient.read()); // Float32Array [2, 4, 6]

result.dispose();
gradient.dispose();
backend.dispose();
```

In Node, initialize WebGPU globals and a device first:

```ts
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter) throw new Error('No WebGPU adapter');
const device = await adapter.requestDevice();
const backend = await typegpu({ device });
// ... compile / run ...
backend.dispose();
device.destroy(); // The caller owns externally supplied devices.
```

Calling a kernel directly executes its ordinary JavaScript body. Tensor operations execute in their inputs' environment: host tensors use the CPU reference implementation, while GPU tensors submit GPU operations. Direct calls execute operation by operation; `compile()` uses kernel IR to fuse operations across the function. Mixing host and device tensors, or tensors belonging to different backends, requires an explicit `await destination.copyFrom(source)` first. JS scalars can be used with tensors in either environment.

Calling an executable submits GPU commands and immediately returns GPU-backed tensors; it does not wait for computation to finish. `read()` and `item()` asynchronously wait for the data and copy it to the CPU. Outputs have independent storage, so later executions do not overwrite them. Dispose them when no longer needed.

Executables lease reusable input, intermediate, and constant buffers. `program.dispose()` returns these buffers to the backend's pool. TypeGPU retains up to 64 MiB of free workspace and up to 32 cached plans for direct operations. Result tensors have independent storage. Disposing the backend also disposes its executables, GPU tensors, and pooled resources.

The common backend exposes `capabilities.dtypes`, `copyFrom()`, `compile()`, `scope()`, `synchronize()`, and `dispose()`. `await backend.synchronize()` waits for submitted work without reading a tensor. Backend-specific diagnostics live in `@blaten/typegpu`: `inspect(program)` reports `dispatchCount` and leased `workspaceBytes`. Dispatch count measures compute dispatches per execution, not queue submissions or elapsed time. One compiled function can submit several dispatches together.

`tensor([[1, 2], [3, 4]])` infers its shape. For TypedArrays, use `tensor(values, { shape: [2, 2] })` to specify a shape. `zeros(shape)`, `ones(shape)`, and `randn(shape, { seed, scale })` are also available.

## Dtypes

Tensors and tensor IR nodes carry a dtype. CPU operations support `bool`, `i8/u8`, `i16/u16`, `i32/u32`, `i64/u64`, and `f16/f32/f64`. Numeric arrays default to f32, boolean arrays to bool, and bigint arrays to i64. TypedArrays retain their storage dtype. An explicit dtype converts the supplied values.

```ts
import { tensor, zeros, i32, i64, f16, f64 } from 'blaten';

const precise = tensor([1 + 2 ** -40], { dtype: f64 });
const indices = tensor(new Int32Array([0, 1, 2]));
const large = tensor([9007199254740993n], { dtype: i64 });
const half = zeros([2, 3], { dtype: f16 });
const mask = indices.gt(0); // bool tensor
const values = mask.cast(i32);
const inverse = mask.not();
```

`Tensor<S, D>.read()` returns a fresh host snapshot with the corresponding TypedArray type. Tensors expose neither a mutable `data` field nor their storage; backend implementations access storage through `blaten/backend`. Bool uses a Uint8Array containing 0 or 1, while `item()` returns a boolean. i64/u64 use BigInt64Array/BigUint64Array and bigint scalars. f16 requires native Float16Array; creating an f16 host tensor reports a feature-support error if it is unavailable.

| Dtype | CPU execution | TypeGPU |
| --- | --- | --- |
| bool, i8/u8, i16/u16, i32/u32 | Supported | Supported; bool and narrow integers use 32-bit storage |
| f16 | Requires Float16Array | Requires the device's shader-f16 feature |
| f32 | Supported | Supported |
| f64, i64/u64 | Supported | Rejected; no implicit precision downgrade |

`backend.capabilities.dtypes` lists the dtypes supported by the device. When `typegpu()` creates a device, it enables shader-f16 if supported by the adapter. For an external device, request shader-f16 through `requiredFeatures` yourself. Native shader scalar types follow the [WGSL specification](https://www.w3.org/TR/WGSL/#scalar-types).

Promotion prefers the wider floating type. Integer promotion widens to accommodate signed/unsigned ranges; mixing i64 and u64 promotes to f64. Weak JS scalar literals adopt a tensor's dtype when representable. Integer overflow wraps at the fixed bit width. Integer division and integer exp/log/tanh/mean/softmax produce f32. Sum preserves numeric dtype, while bool sum produces u32. f16 matmul, reductions, and softmax accumulate internally in f32. These rules do not aim to reproduce every NumPy or PyTorch promotion rule.

Bool arithmetic requires an explicit numeric cast; and/or/not accept only bool inputs. Differentiation requires floating outputs and targets. Casts between f16/f32/f64 pass gradients; integer and bool paths do not. Registered map callbacks specialize to their input dtype, keeping intermediate and result dtypes consistent across CPU and GPU.

`tensorType(shape, dtype)` is a low-level type value used by backend IR lowering. Application `compile()` calls do not take it. GPU execution validates both input shapes and dtypes.

## Differentiation

`grad(fn, { arg: 0 })` and `valueAndGrad(fn, { arg: [2, 3] })` construct derivative definitions without executing the function. The original function must return a scalar-shaped value (`[]`). Derivatives can be nested, as in `grad(grad(fn))`. The default differentiation target is the first argument. Targets may be tensors, fixed tuples, or nested plain records; gradients preserve that structure. Multiple selected arguments produce an array of gradient trees (a single selected argument produces its gradient tree). `valueAndGrad` returns `{ value, grads }`; this replaces the earlier `[value, ...gradients]` result. Tensor objects do not carry gradient state.

A custom VJP receives the original inputs followed by the output cotangent, and returns one gradient per input. Rules are kernels too, so differentiating the rules supports higher derivatives.

```ts
import { customVjp, customJvp } from 'blaten';

function square(x: number) { 'use kernel'; return x * x; }
function backward(x: number, dy: number) {
  'use kernel';
  return [2 * x * dy];
}
const withVjp = customVjp(square, { backward });

function tangent(x: number, dx: number) {
  'use kernel';
  return 2 * x * dx;
}
const withJvp = customJvp(square, { jvp: tangent });
```

JVP arguments are `(...inputs, ...inputTangents)`. Its output must match the original output shape. The implementation conservatively checks homogeneous linearity in the tangent, then transposes that linear map for reverse-mode differentiation. Generated derivatives must be compiled before execution.

Keep training parameters on the GPU with `await backend.copyFrom(tensorOrTree)`. Fixed arrays and nested plain records of tensors are copied with their structure intact. `copyFrom` preserves shape and dtype and always returns independent storage owned by the destination backend, leaving the source intact. Host inputs are copied to the device; copies within the same backend stay on the GPU. Other backends currently use a host snapshot as an intermediate. The Promise resolves after copying has been submitted, without waiting for GPU completion; `read()` or `item()` waits when host values are needed.

SGD belongs to the common runtime and uses tensor IR in the parameters' environment:

```ts
import { sgd } from 'blaten';
const optimizer = sgd(parameters, { learningRate: 0.1 });
optimizer.step(gradients);
```

Parameters and gradients must have matching tree structure, environment, shapes, and floating dtypes. Record key insertion order does not matter. Parameter tensors must be distinct. Updates mutate parameter storage without reading GPU gradients on the CPU. The MNIST example normalizes data and prepares batches on the CPU, performs forward/backward computation and parameter updates on the GPU, and accumulates scalar loss on the GPU. Training batches submit work without a host read; epoch reporting reads average loss once. Evaluation accumulates multiclass accuracy on the GPU, excludes padded rows, and reads one scalar at the evaluation boundary. The example's `src/train.ts` contains training and evaluation; `src/run.ts` handles execution settings, dataset loading, result metadata, saving, and learning verification.

## Training and tensor lifetimes

Put parameters in the first argument of a kernel and compile its value and gradient together:

```ts
type Parameters = { weights: Tensor<[784, 10], typeof f32>; bias: Tensor<[10], typeof f32> };
function loss(p: Parameters, x: Tensor<[number, 784], typeof f32>, y: Tensor<[number, 10], typeof f32>) {
  'use kernel';
  return crossEntropy(x.matmul(p.weights).add(p.bias), y);
}
const parameters = await backend.copyFrom(initialParameters);
const training = await backend.compile(valueAndGrad(loss), { inputs: [parameters, sampleX, sampleY] });
using optimizer = accumulate(sgd(parameters, { learningRate: 0.1 }), { steps: 4 });
using lossMetric = meanMetric();

for (const batch of epochBatches) {
  backend.scope(() => {
    const [x, y] = batch.toTensors();
    const { value, grads } = training(parameters, x, y);
    optimizer.step(grads, { weight: batch.size });
    lossMetric.add(value, { weight: batch.size });
  });
}
optimizer.flush();
const lossValue = await backend.scope(() => lossMetric.compute().item());
```

`scope()` tracks tensors created synchronously inside its callback, including CPU batch tensors and GPU operation results. It releases temporaries on normal return or exception. Returned tensors, including tensors inside arrays or records, escape to the caller (or an enclosing scope); borrowed input tensors are untouched. A synchronous callback can return a read Promise: cleanup waits for it to settle, including rejection. Async callbacks are rejected; allocations after an `await` or inside a Promise continuation are not part of the scope. This keeps allocation ownership local without a Node-specific async context. `scope()` is also exported directly by `blaten`, and `disposeTree()` releases an explicitly owned tensor tree.

`accumulate(optimizer, { steps })` wraps any `Optimizer<P>` exposing its bound `parameters` and `step(gradients)`. It borrows that optimizer and its parameters and owns the accumulation buffers. `step(grads, { weight })` averages gradients using positive finite weights and applies the wrapped optimizer every `steps` calls. `steps: 1` applies gradients directly. `flush()` applies a final partial window and is a no-op when empty. Disposal discards pending gradients; it does not apply updates or dispose the wrapped optimizer or parameters.

`meanMetric({ dtype: f32 })` borrows floating scalar tensors and accumulates their weighted sum in the first value's environment. The default accumulation dtype is f32. `add(value, { weight })` performs no host reads; values must share an environment and weights must be positive and finite. `compute()` returns an independently owned scalar tensor, and rejects an empty metric. `reset()` releases accumulated state for the next epoch without invalidating computed tensors. Read host values explicitly with `item()` or `read()` on the result.

`argmax(x, axis)` / `x.argmax(axis)` return i32 indices with the selected axis removed (negative axes are supported). Ties choose the first index; if there are NaNs, the first NaN is selected. `eq(a, b)` / `a.eq(b)` compare values after dtype promotion and return a bool tensor with the broadcast shape. Scalar-only `eq` returns a JS boolean. Both operations are nondifferentiable.

`multiclassAccuracy()` implements `Accumulator<MulticlassAccuracyInput, TensorF32<[]>>`. Pass one object to `add`:

```ts
using accuracy = multiclassAccuracy();
backend.scope(() => {
  const [x, y] = batch.toTensors();
  accuracy.add({ logits: inference(parameters, x), targets: y, validCount: batch.size });
});
const score = await backend.scope(() => accuracy.compute().item());
```

Logits and one-hot targets must have matching `[batch, classes]` shapes. `validCount` defaults to the full batch and selects the leading real rows, excluding padding; it must be a positive integer at most the batch size. The metric accumulates correct and sample counts in one environment, and `compute()` returns their ratio as an independently owned f32 scalar without a host read. When an input is on the GPU, host inputs are uploaded to that device; inputs from different devices are rejected. Optional common accumulator weights uniformly weight the samples in each added batch. `weight` reports the total weighted sample count. `reset()` releases the metric state and computed snapshots remain valid.

Tensors, executables (including lazy functions and prepared leases), gradient accumulators, accumulating optimizer wrappers, and metrics implement `Symbol.dispose` as well as `dispose()`, so TypeScript `using` declarations release them on block exit, including exceptions. Persistent helper buffers survive batch scopes and are released by their owning helper.

Device tensors and executable functions also register independent resource state with `FinalizationRegistry`. Backends track resources rather than retaining those objects, so dropping the last reference allows GC cleanup. Finalizers are a fallback: their execution and timing are not guaranteed, and GPU memory pressure does not guarantee JavaScript GC. Use `using`, `dispose()`, or batch `scope()` for predictable memory bounds. Explicit disposal unregisters the finalizer; backend disposal invalidates live tensors and executables and releases all remaining resources. An asynchronous tensor read keeps its tensor alive until the read finishes. Backend storage and cleanup callbacks must never retain their owning Tensor or Executable.

All mean accumulators implement the public `Accumulator<Input, Output = Input>` interface: `weight`, `add(value, { weight })`, `compute()`, `reset()`, `dispose()`, and `Symbol.dispose`. The input and result types are independent: scalar metrics accept any floating scalar dtype and return their configured dtype, while tree accumulators preserve the template's shape, dtype, tuple, and record types.

`meanAccumulator(template)` computes a weighted mean of any nonempty floating tensor tree, including vectors and nested records. The template supplies only structure, shape, dtype, and environment metadata; its values are not accumulated and it may be disposed after construction. Inputs must match that metadata. Repeated tensor leaves are allowed. `reset()` zeros and reuses persistent buffers, including after nonfinite inputs, without invalidating computed snapshots. `gradientAccumulator(parameters)` is a thin wrapper that additionally requires distinct parameter tensors. Both use `compute()` and `reset()`; the earlier `mean()` and `clear()` names have been removed.

For manual accumulation, keep an accumulator outside the batch scope:

```ts
const accumulator = gradientAccumulator(parameters);
backend.scope(() => {
  const { grads } = training(parameters, x, y);
  accumulator.add(grads, { weight: sampleCount });
});
// Repeat add() for more batches before updating parameters.
backend.scope(() => {
  optimizer.step(accumulator.compute());
  accumulator.reset();
});
accumulator.dispose();
```

`add()` borrows gradients and maintains independent persistent buffers in their environment. On GPU parameters, weighting, addition, averaging, clearing, and SGD stay on the GPU. `weight` is a positive finite number (default 1); use the actual sample count for batches whose loss is a per-sample mean. `compute()` returns independently owned tensors and rejects an empty accumulator. `reset()` resets the weight and reuses accumulation buffers; `dispose()` releases them. Flush a final partial accumulation window using its actual weight.

## Function types and compilation

Shape is the first type parameter: `Tensor<S = Shape, D = DType>`. You can write `Tensor<[2, 3], typeof f32>` without `readonly`; the shape property is `Readonly<S>` and is frozen at runtime. Public dtype aliases are `TensorBool`, `TensorI8`, `TensorU8`, `TensorI16`, `TensorU16`, `TensorI32`, `TensorU32`, `TensorI64`, `TensorU64`, `TensorF16`, `TensorF32`, and `TensorF64`. Each accepts `S extends Shape = Shape` and fixes the corresponding dtype: use `TensorF32<[2, 3]>` for a known shape or `TensorF32` for an arbitrary shape. These aliases have no runtime cost. The rank type is `S['length']`. Creation functions default to f32 for numeric data, while an unspecified dtype in the Tensor type means DType.

Tensor methods and exported operation functions preserve known shapes and dtypes through chains. Broadcasting, rank-2 matmul, transpose, reshape, and reductions infer their resulting shapes; reductions support literal axes (including negative axes), axis tuples, and `keepDims`. Dynamic shapes or axes fall back to `Shape`. Mixed tensor dtypes follow promotion rules. Integer tensors combined with JS numeric scalars retain a union of possible dtypes because scalar adoption depends on whether the runtime value fits the integer range.

The build plugin extracts input signatures from TypeScript parameter types, resolving imported aliases and `typeof` dtype constants. Numbers lower to f32, booleans to bool, and bigints to i64. Tensor inputs need fixed shape tuples and a single concrete dtype to compile directly. Fixed tuples and records of these input types are supported recursively; optional fields, sparse arrays, class instances, and recursive record types are not supported. Kernels can access fixed record fields and return nested tensor records or arrays.

`compile(fn, backend)` uses a concrete extracted signature. For unknown dimensions or dtype unions, use `compile(fn, backend, { inputs: [...] })`. Compilation derives actual shapes and dtypes and validates them against declared constraints, including fixed dimensions. Untyped or unsupported parameter types still produce a runtime error. Example inputs supply types; their tensor values are not captured. Output shapes and dtypes are inferred from tensor IR; return-type annotations are not enforced as output constraints. Input and result types are preserved by the callable executable, so ordinary application calls do not need result casts.

`compileLazy(fn, backend, { cacheSize: 8 })` provides lazy asynchronous calls. It specializes on the first call for each tree layout, shape, and dtype and maintains an LRU cache (default eight plans); scalar values are runtime inputs, not cache keys. Use `await lazy(...inputs)` for dynamic signatures. For scoped synchronous execution, `using program = await lazy.prepare(...exampleInputs)` returns a callable lease for use inside `scope()`. Prepared leases remain valid across cache eviction and lazy-function disposal or GC; they retain their individual plans without retaining the cache. `lazy.dispose()` invalidates new calls and releases cached plans once active leases finish. Disposing or collecting the last lease releases a retired plan. Backend disposal still invalidates every plan. Compilation failures are not cached. Explicit `compile()` remains available when startup preparation is preferable.

```ts
function genericSum(x: Tensor) {
  'use kernel';
  return x.mul(x).sum();
}
function sum784(x: Tensor<[784], typeof f32>) {
  'use kernel';
  return genericSum(x);
}
const program = await compile(sum784, backend);
const specialized = await compile(genericSum, backend, {
  inputs: [tensor([1, 2, 3])],
});
```

A compiled program accepts only its specialized shapes and dtypes. Compile another program for another shape; automatic recompilation and a single program with dynamic shapes are not implemented.

## IR and backend boundaries

```text
TypeScript + 'use kernel'
  -> @blaten/compiler/vite: AST -> kernel IR, TS types -> input constraints/signature
  -> compile: kernel IR + constraints + optional example inputs -> tensor IR
  -> grad / custom rules: differentiation of tensor IR
  -> TypeGPU backend: fusion, execution planning, tinyest generation
  -> TypeGPU: WGSL generation, resource binding, WebGPU pipelines
```

Build tooling consumes application ASTs. Metadata v2 holds kernel IR, input constraints, an optional concrete signature or extraction-failure reason, and external-reference getters; it contains neither source text nor compiler AST/type objects. Automatic differentiation works on shape-resolved tensor IR. Captured host tensors become compile-time constant snapshots; pass device tensors as explicit inputs.

Application functions are not registered with TypeGPU. Only generated compute functions' tinyest ASTs are passed to TypeGPU. This connection is isolated in [adapter.ts](packages/typegpu/src/adapter.ts), using the private `__TYPEGPU_META__` v2 contract. The peer range is **~0.12.7** and tinyest is pinned to **0.3.4**. Revalidate that contract before widening compatibility to TypeGPU 0.13.0 or later.

Kernel definitions are stored on their function objects under `Symbol.for('blaten.kernel')`, exported as `kernelIr` by `blaten` and `blaten/internal`. `registerKernel()` attaches ordinary kernels; grad, valueAndGrad, customVjp, and customJvp attach definitions to their generated functions using the same symbol. The property is non-enumerable, non-writable, and non-configurable. Different copies of the runtime in the same JavaScript realm can read it. `definitionOf()` reads only own properties and validates the IR version. Ordinary kernel IR is available at `definitionOf(fn).metadata.ir`. Binding a function or wrapping it arbitrarily does not copy its definition.

`blaten/ir` exposes kernel IR types: a linear value graph of parameters, external references, literals, arrays, records, arithmetic, member access, and calls. Local variables and TS syntax disappear during the build. Helpers and map callbacks remain function references. Shape reads and external values resolve during runtime lowering. Concretely typed wrappers specialize generic helpers. The build plugin validates source syntax; runtime lowering validates shapes, operations, and call targets.

The core's [lowering.ts](packages/blaten/src/lowering.ts) specializes kernel IR into tensor IR. Legacy AST metadata v1 is not accepted; update and rebuild compiler and runtime together. `compile(fn, backend)` delegates to the backend's compile method. Future native/WASM backends can use the same [runtime interface](packages/blaten/src/runtime.ts).

## Current scope

- Kernel bodies support initialized const declarations, returns, arithmetic, fixed arrays, shape references, registered operations, and other kernel calls.
- Operations include broadcast arithmetic, exp/log/tanh/ReLU, rank-2 matmul/transpose, reshape, sum/mean, dtype casts, gt comparisons, bool and/or/not, last-axis softmax, one-hot cross entropy, and scalar-kernel map.
- Called helpers and map callbacks also require `'use kernel'`. Module-level declarations and function expressions are supported. Built-in operations are registered intrinsics.
- Elementwise/view operations fuse into their consumers. Matmul, reductions, and softmax can materialize GPU intermediates. Compiling a whole training function does not imply one compute dispatch.
- Branches, loops, recursion, mutation, arbitrary JS calls, dynamic shapes, empty tensors, and complex dtypes are not supported.
- Direct CPU execution is a reference JS implementation. SIMD/BLAS optimization and native/WASM backends are not implemented.
- GPU matmul and reduction kernels use simple loops without shared-memory tiling or architecture-specific tuning.

Inspect the operation registry through `blaten/intrinsics`. `intrinsics.ts` contains both implementations and their single registration definitions; the linter and runtime lowering share them. `getIntrinsics()` returns the current list without memoization. `getIntrinsic(nameOrFunction)` finds the same definition by name or function identity. Tensor-method availability is inspected from Tensor.prototype. Each definition includes the public function, tensor-method availability, and zero-based kernel callback argument positions (`map` has `[0]`). Adding an intrinsic does not require a separate linter allowlist.

```ts
import { getIntrinsics } from 'blaten/intrinsics';
console.log(getIntrinsics().map(({ name }) => name));
const methods = getIntrinsics().filter(({ tensorMethod }) => tensorMethod);
```

The ESLint plugin checks unsupported kernel syntax and unmarked helpers, including directly exported functions imported through relative paths. Its external helper resolution is currently limited to direct exports; runtime lowering validates the actual registered definitions again.

```js
import tseslint from 'typescript-eslint';
import blaten from '@blaten/eslint-plugin';

export default [{
  files: ['**/*.ts'],
  languageOptions: { parser: tseslint.parser },
  plugins: { blaten },
  rules: { 'blaten/kernel-contract': 'error' },
}];
```

`pnpm test` uses a real WebGPU device to verify fusion, shape validation, imported helpers, tensor methods, finite-difference checks for broadcasting/matmul/softmax gradients, fifth derivatives, second derivatives of custom VJP/JVP rules, GPU SGD, and the linter. It also checks CPU dtype behavior, symbol metadata, and the build-time AST-to-IR and type-to-signature boundary.
