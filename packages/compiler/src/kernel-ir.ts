import type { KernelIR, KernelInstruction } from 'blaten/ir';

/** Resolve lexical bindings into value references; no source AST crosses the build boundary. */
export function toKernelIR(fn: any, externals: ReadonlySet<string>): KernelIR {
  const instructions: KernelInstruction[] = [];
  const bindings = new Map<string, number>();
  const emit = (instruction: KernelInstruction): number => instructions.push(instruction) - 1;
  fn.params.forEach((parameter: any, index: number) => bindings.set(parameter.name, emit({ op: 'argument', index })));
  const externalValues = new Map<string, number>();
  const expression = (node: any): number => {
    if (!node) throw new Error('Kernel expressions must have a value');
    switch (node.type) {
      case 'TSAsExpression': case 'TSNonNullExpression': return expression(node.expression);
      case 'Identifier': {
        const local = bindings.get(node.name);
        if (local !== undefined) return local;
        if (!externals.has(node.name)) throw new Error(`Unknown kernel binding ${node.name}`);
        let value = externalValues.get(node.name);
        if (value === undefined) { value = emit({ op: 'external', name: node.name }); externalValues.set(node.name, value) }
        return value;
      }
      case 'NumericLiteral': case 'BooleanLiteral': case 'StringLiteral': return emit({ op: 'literal', value: node.value });
      case 'BigIntLiteral': return emit({ op: 'bigint', value: node.value });
      case 'ArrayExpression': return emit({ op: 'array', values: node.elements.map(expression) });
      case 'BinaryExpression': {
        const operators = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '**': 'pow', '>': 'gt' } as const;
        const operator = operators[node.operator as keyof typeof operators];
        if (!operator) throw new Error(`Unsupported kernel operator ${node.operator}`);
        return emit({ op: 'binary', operator, left: expression(node.left), right: expression(node.right) });
      }
      case 'UnaryExpression':
        if (node.operator !== '-') throw new Error('Unsupported kernel unary operator');
        return emit({ op: 'neg', value: expression(node.argument) });
      case 'MemberExpression': return emit({ op: 'member', object: expression(node.object), key: node.computed ? expression(node.property) : emit({ op: 'literal', value: node.property.name }) });
      case 'CallExpression': return emit({ op: 'call', target: expression(node.callee), args: node.arguments.map(expression) });
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
      return { parameterCount: fn.params.length, instructions, result: expression(statement.argument) };
    } else throw new Error(`Unsupported kernel statement ${statement.type}. Use const bindings and return.`);
  }
  throw new Error('Kernel has no return statement');
}
