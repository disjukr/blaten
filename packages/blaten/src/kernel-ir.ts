/** Shape-independent, linear value graph emitted by the build compiler. */
export interface KernelIR {
  readonly parameterCount: number;
  readonly instructions: readonly KernelInstruction[];
  readonly result: number;
}
/** Value references are zero-based indices into the instruction list. */
export type KernelInstruction =
  | { readonly op: 'argument'; readonly index: number }
  | { readonly op: 'external'; readonly name: string }
  | { readonly op: 'literal'; readonly value: number | boolean | string }
  | { readonly op: 'bigint'; readonly value: string }
  | { readonly op: 'array'; readonly values: readonly number[] }
  | { readonly op: 'binary'; readonly operator: 'add' | 'sub' | 'mul' | 'div' | 'pow' | 'gt'; readonly left: number; readonly right: number }
  | { readonly op: 'neg'; readonly value: number }
  | { readonly op: 'member'; readonly object: number; readonly key: number }
  | { readonly op: 'call'; readonly target: number; readonly args: readonly number[] };
