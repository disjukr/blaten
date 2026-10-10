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

MNIST downloads the [CVDF IDX files](https://storage.googleapis.com/cvdf-datasets/mnist/) on first use and caches them in `examples/mnist/data/`. Defaults are 12,000 training images, 2,000 separate test images, batch size 64, and three epochs. Use `TRAIN_SIZE=60000` and `TEST_SIZE=10000` for the full dataset. `EPOCHS` and `LEARNING_RATE` are also configurable through environment variables.

The model's `Images` and `Labels` types leave the batch dimension as `number`. `kernels.ts` sets `batchSize = 64` and specializes the compilation entry points with `Images<typeof batchSize>` and `Labels<typeof batchSize>`. `model.ts` sets `hiddenSize = 32`. Change these constants and recompile to use other sizes. Results are written to `examples/mnist/results/latest.json`.

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
import { typegpu } from '@blaten/typegpu';

function squaredSum(x: Tensor<[3], typeof f32>) {
  'use kernel';
  return x.mul(x).sum();
}

// Uses navigator.gpu in a browser.
const backend = await typegpu();
const input = tensor([1, 2, 3]);
const program = await compile(squaredSum, backend);
const result = program.run(input) as Tensor<Shape, typeof f32>;
console.log(await result.item()); // 14
console.log(program.dispatchCount); // 1: multiply and sum share a shader

const derivative = await compile(grad(squaredSum), backend);
const gradient = derivative.run(input) as Tensor<Shape, typeof f32>;
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

Calling a kernel directly executes ordinary JavaScript on the CPU. For example, `squaredSum(tensor([1, 2, 3], { shape: [3] }))` computes on the CPU. `compile()` specializes kernel IR using the function's registered input signature and creates a program for fixed input shapes and dtypes.

`run()` submits GPU commands and immediately returns GPU-backed tensors; it does not wait for computation to finish. `read()` and `item()` asynchronously wait for the data and copy it to the CPU. Outputs have independent storage, so later executions do not overwrite them. Dispose them when no longer needed.

An executable owns reusable input, intermediate, and constant buffers. `program.dispose()` releases those buffers; disposing the backend also disposes its remaining executables and GPU tensors. `dispatchCount` reports compute dispatches per execution, not queue submissions or elapsed GPU time. One compiled function can contain several dispatches submitted together.

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

The `data` and `read()` types of `Tensor<S, D>` reflect the corresponding TypedArray. Bool uses a Uint8Array containing 0 or 1, while `item()` returns a boolean. i64/u64 use BigInt64Array/BigUint64Array and bigint scalars. f16 requires native Float16Array; creating an f16 host tensor reports a feature-support error if it is unavailable.

| Dtype | CPU execution | TypeGPU |
| --- | --- | --- |
| bool, i8/u8, i16/u16, i32/u32 | Supported | Supported; bool and narrow integers use 32-bit storage |
| f16 | Requires Float16Array | Requires the device's shader-f16 feature |
| f32 | Supported | Supported |
| f64, i64/u64 | Supported | Rejected; no implicit precision downgrade |

`backend.supportedDTypes` lists the dtypes supported by the device. When `typegpu()` creates a device, it enables shader-f16 if supported by the adapter. For an external device, request shader-f16 through `requiredFeatures` yourself. Native shader scalar types follow the [WGSL specification](https://www.w3.org/TR/WGSL/#scalar-types).

Promotion prefers the wider floating type. Integer promotion widens to accommodate signed/unsigned ranges; mixing i64 and u64 promotes to f64. Weak JS scalar literals adopt a tensor's dtype when representable. Integer overflow wraps at the fixed bit width. Integer division and integer exp/log/tanh/mean/softmax produce f32. Sum preserves numeric dtype, while bool sum produces u32. f16 matmul, reductions, and softmax accumulate internally in f32. These rules do not aim to reproduce every NumPy or PyTorch promotion rule.

Bool arithmetic requires an explicit numeric cast; and/or/not accept only bool inputs. Differentiation requires floating outputs and targets. Casts between f16/f32/f64 pass gradients; integer and bool paths do not. Registered map callbacks specialize to their input dtype, keeping intermediate and result dtypes consistent across CPU and GPU.

`tensorType(shape, dtype)` is a low-level type value used by backend IR lowering. Application `compile()` calls do not take it. GPU execution validates both input shapes and dtypes.

## Differentiation

`grad(fn, { arg: 0 })` and `valueAndGrad(fn, { arg: [2, 3] })` construct derivative definitions without executing the function. The original function must return a scalar-shaped value (`[]`). Derivatives can be nested, as in `grad(grad(fn))`. Gradients for multiple arguments are returned as an array; `valueAndGrad` returns `[value, ...gradients]`.

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

Keep training parameters on the GPU with `backend.upload(tensor)` and update them with `backend.sgd(parameters, gradients, learningRate)`. Gradients need not be read on the CPU. The MNIST example normalizes data and prepares batches on the CPU, performs forward/backward computation and parameter updates on the GPU, and reads scalar loss for logging.

## Function types and compilation

Shape is the first type parameter: `Tensor<S = Shape, D = DType>`. You can write `Tensor<[2, 3], typeof f32>` without `readonly`; the shape property is `Readonly<S>` and is frozen at runtime. The rank type is `S['length']`. Creation functions default to f32 for numeric data, while an unspecified dtype in the Tensor type means DType.

The build plugin extracts input signatures from TypeScript parameter types, resolving imported aliases and `typeof` dtype constants. Numbers lower to f32, booleans to bool, and bigints to i64. Tensor inputs need fixed shape tuples and a single concrete dtype to compile directly.

`compile(fn, backend)` takes no separate signature argument. Missing or nonconcrete input signatures cause a runtime error. A generic helper can still be called inside a concretely typed entry point. Output shapes and dtypes are inferred from tensor IR; return-type annotations are not enforced as output constraints.

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
// compile(genericSum, backend) rejects: no concrete input signature.
```

Automatic specialization from actual inputs and dynamic-shape execution are not implemented. A compiled program accepts only its fixed shapes and dtypes.

## IR and backend boundaries

```text
TypeScript + 'use kernel'
  -> @blaten/compiler/vite: AST -> kernel IR, TS types -> input signature
  -> compile: kernel IR + attached signature -> shape-resolved tensor IR
  -> grad / custom rules: differentiation of tensor IR
  -> TypeGPU backend: fusion, execution planning, tinyest generation
  -> TypeGPU: WGSL generation, resource binding, WebGPU pipelines
```

Build tooling consumes application ASTs. Metadata v2 holds kernel IR, an input signature or extraction-failure reason, and external-reference getters; it contains neither source text nor compiler AST/type objects. Automatic differentiation works on shape-resolved tensor IR.

Application functions are not registered with TypeGPU. Only generated compute functions' tinyest ASTs are passed to TypeGPU. This connection is isolated in [adapter.ts](packages/typegpu/src/adapter.ts), using the private `__TYPEGPU_META__` v2 contract. The peer range is **~0.12.7** and tinyest is pinned to **0.3.4**. Revalidate that contract before widening compatibility to TypeGPU 0.13.0 or later.

Kernel definitions are stored on their function objects under `Symbol.for('blaten.kernel')`, exported as `kernelIr` by `blaten` and `blaten/internal`. `registerKernel()` attaches ordinary kernels; grad, valueAndGrad, customVjp, and customJvp attach definitions to their generated functions using the same symbol. The property is non-enumerable, non-writable, and non-configurable. Different copies of the runtime in the same JavaScript realm can read it. `definitionOf()` reads only own properties and validates the IR version. Ordinary kernel IR is available at `definitionOf(fn).metadata.ir`. Binding a function or wrapping it arbitrarily does not copy its definition.

`blaten/ir` exposes kernel IR types: a linear value graph of parameters, external references, literals, arrays, arithmetic, member access, and calls. Local variables and TS syntax disappear during the build. Helpers and map callbacks remain function references. Shape reads and external values resolve during runtime lowering. Concretely typed wrappers specialize generic helpers. The build plugin validates source syntax; runtime lowering validates shapes, operations, and call targets.

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
