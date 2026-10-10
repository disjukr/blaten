import { fileURLToPath } from 'node:url';
import { ESLint } from 'eslint';
import tseslint from 'typescript-eslint';
import lintPlugin from '@blaten/eslint-plugin';
import * as blaten from 'blaten';
import { getIntrinsic, getIntrinsics } from 'blaten/intrinsics';
import assert from 'node:assert/strict';
const eslint = new ESLint({ overrideConfigFile: true, overrideConfig: [{ files: ['**/*.ts'], languageOptions: { parser: tseslint.parser }, plugins: { blaten: lintPlugin }, rules: { 'blaten/kernel-contract': 'error' } }] });
const filename = fileURLToPath(new URL('contract.ts', import.meta.url));
const intrinsics = getIntrinsics();
assert.equal(blaten.getIntrinsics, getIntrinsics, 'Root and tooling entry points expose the same API');
assert.equal(blaten.getIntrinsic, getIntrinsic);
assert.equal(getIntrinsic(blaten.tensor), undefined);
assert.equal(getIntrinsic('read'), undefined);
assert.equal(new Set(intrinsics.map(op => op.name)).size, intrinsics.length);
assert.ok(intrinsics.length > 0);
assert.ok(!intrinsics.some(op => op.fn === blaten.tensor || op.fn === blaten.grad || op.fn === blaten.compile), 'Runtime utilities are not kernel intrinsics');
for (const op of intrinsics) {
  assert.equal(getIntrinsic(op.name), op, 'Name lookup shares the registered definition');
  assert.equal(getIntrinsic(op.fn), op, 'Function lookup shares the registered definition');
  assert.equal(blaten[op.name], op.fn, `${op.name} must be a real public operation`);
  assert.equal(op.tensorMethod, typeof Object.getOwnPropertyDescriptor(blaten.Tensor.prototype, op.name)?.value === 'function');
  const args = ['x'];
  for (const index of op.kernelParameters) args[index] = 'square';
  const source = `import { ${op.name} as operation } from 'blaten'; function square(x: number) { 'use kernel'; return x*x } function f(x: any) { 'use kernel'; return operation(${args.join(', ')}) }`;
  const [named] = await eslint.lintText(source, { filePath: filename });
  assert.equal(named.errorCount, 0, `${op.name}: ${JSON.stringify(named.messages)}`);
  const [method] = await eslint.lintText(`function f(x: any) { 'use kernel'; return x.${op.name}(x) }`, { filePath: filename });
  assert.equal(method.errorCount, op.tensorMethod ? 0 : 1, `${op.name} method: ${JSON.stringify(method.messages)}`);
}
for (const [source, expected] of [
  [`import { sum, map as each, type Tensor } from 'blaten'; import { square } from './helper.js'; function f(x: Tensor) { 'use kernel'; return sum(each(square, x)) }`, 0],
  [`function f(x: number) { return x * 2 } function g(x: number) { 'use kernel'; return f(x) }`, 1],
  [`import { map as each } from 'blaten'; function f(x: number) { return x*x } function g(x: any) { 'use kernel'; return each(f,x) }`, 1],
  [`function f(x: number) { 'use kernel'; if (x > 0) return x; return -x }`, 1],
  [`function f(x: any) { 'use kernel'; return x.mul(x).sum() }`, 0],
  [`import { customVjp } from 'blaten'; function f(x: number) { 'use kernel'; return x*x } const wrapped=customVjp(f, {}); function g(x: number) { 'use kernel'; return wrapped(x) }`, 0],
  [`import { map as each } from 'blaten'; function f() { 'use kernel'; return each() }`, 1],
  [`import type { mul } from 'blaten'; function f(x: any) { 'use kernel'; return mul(x,x) }`, 1],
  [`import { type mul } from 'blaten'; function f(x: any) { 'use kernel'; return mul(x,x) }`, 1],
  [`import { tensor } from 'blaten'; function f() { 'use kernel'; return tensor([1]) }`, 1],
  [`function f(x: any) { 'use kernel'; return x.read() }`, 1],
  [`import { map } from './helper.js'; function f(x: any) { 'use kernel'; return map(x) }`, 0],
]) {
  const [result] = await eslint.lintText(source, { filePath: filename });
  assert.equal(result.errorCount, expected, JSON.stringify(result.messages));
}
console.log('Passed: discovered intrinsic API, every intrinsic import and Tensor method, kernel callback contracts, and imported helpers.');
