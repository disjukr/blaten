import type { KernelIR, KernelInstruction } from 'blaten/ir';

/** Resolve lexical bindings into value references; no source AST crosses the build boundary. */
export function toKernelIR(fn: any, externals: ReadonlySet<string>, trace = false): KernelIR {
  const instructions: KernelInstruction[] = [];
  const spans: { start: number; end: number }[][] = [];
  const bindings = new Map<string, number>();
  const locate = (index: number, node: any): number => {
    if (trace && Number.isInteger(node?.start) && Number.isInteger(node?.end)) {
      const locations = spans[index]!;
      if (!locations.some(s => s.start === node.start && s.end === node.end)) locations.push({ start: node.start, end: node.end });
    }
    return index;
  };
  const emit = (instruction: KernelInstruction, node: any): number => {
    const index = instructions.push(instruction) - 1;
    if (trace) spans.push([]);
    return locate(index, node);
  };
  fn.params.forEach((parameter: any, index: number) => bindings.set(parameter.name, emit({ op: 'argument', index }, parameter)));
  const externalValues = new Map<string, number>();
  const expression = (node: any): number => {
    if (!node) throw new Error('Kernel expressions must have a value');
    switch (node.type) {
      case 'TSAsExpression': case 'TSNonNullExpression': return expression(node.expression);
      case 'Identifier': {
        const local = bindings.get(node.name);
        if (local !== undefined) return locate(local, node);
        if (!externals.has(node.name)) throw new Error(`Unknown kernel binding ${node.name}`);
        let value = externalValues.get(node.name);
        if (value === undefined) { value = emit({ op: 'external', name: node.name }, node); externalValues.set(node.name, value) }
        return locate(value, node);
      }
      case 'NumericLiteral': case 'BooleanLiteral': case 'StringLiteral': return emit({ op: 'literal', value: node.value }, node);
      case 'BigIntLiteral': return emit({ op: 'bigint', value: node.value }, node);
      case 'ObjectExpression': {
        const fields: Record<string, number> = {};
        for (const property of node.properties) {
          if (property.type !== 'ObjectProperty' || property.computed || property.method) throw new Error('Kernel objects require fixed properties');
          const key = property.key.name ?? property.key.value;
          if (key === '__proto__') throw new Error('Unsupported kernel object key');
          fields[key] = expression(property.value);
        }
        return emit({ op: 'object', fields }, node);
      }
      case 'ArrayExpression': return emit({ op: 'array', values: node.elements.map(expression) }, node);
      case 'BinaryExpression': {
        const operators = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '**': 'pow', '>': 'gt' } as const;
        const operator = operators[node.operator as keyof typeof operators];
        if (!operator) throw new Error(`Unsupported kernel operator ${node.operator}`);
        return emit({ op: 'binary', operator, left: expression(node.left), right: expression(node.right) }, node);
      }
      case 'UnaryExpression':
        if (node.operator !== '-') throw new Error('Unsupported kernel unary operator');
        return emit({ op: 'neg', value: expression(node.argument) }, node);
      case 'MemberExpression': return emit({ op: 'member', object: expression(node.object), key: node.computed ? expression(node.property) : emit({ op: 'literal', value: node.property.name }, node.property) }, node);
      case 'CallExpression': return emit({ op: 'call', target: expression(node.callee), args: node.arguments.map(expression) }, node);
      default: throw new Error(`Unsupported kernel expression ${node.type}`);
    }
  };
  for (const statement of fn.body.body) {
    if (statement.type === 'VariableDeclaration' && statement.kind === 'const') {
      for (const declaration of statement.declarations) {
        if (declaration.id.type !== 'Identifier' || !declaration.init) throw new Error('Kernel locals must be initialized const identifiers');
        bindings.set(declaration.id.name, expression(declaration.init));
      }
    } else if (statement.type === 'ReturnStatement') {
      return { parameterCount: fn.params.length, instructions, result: expression(statement.argument), ...(trace ? { spans } : {}) };
    } else throw new Error(`Unsupported kernel statement ${statement.type}. Use const bindings and return.`);
  }
  throw new Error('Kernel has no return statement');
}
