import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseSync } from '@babel/core';
import { getIntrinsics, type Intrinsic } from 'blaten/intrinsics';
import type { Rule } from 'eslint';

const intrinsics = new Map(getIntrinsics().map(op => [op.name, op]));
const tensorMethods = new Map([...intrinsics].filter(([, op]) => op.tensorMethod));
const directive = (fn: any) => fn?.body?.type === 'BlockStatement' && (fn.body.body.some((s: any) => s.type === 'ExpressionStatement' && s.expression?.value === 'use kernel') || fn.body.directives?.some((s: any) => s.value.value === 'use kernel'));
function importedKernel(filename: string, source: string, name: string): boolean {
  if (!source.startsWith('.')) return false;
  const base = resolve(dirname(filename), source);
  const path = [base.replace(/\.js$/, '.ts'), base, `${base}.ts`, `${base}.js`].find(existsSync);
  if (!path) return false;
  const parsed = parseSync(readFileSync(path, 'utf8'), { filename: path, configFile: false, babelrc: false, parserOpts: { plugins: ['typescript'], sourceType: 'module' } });
  return parsed?.program.body.some((s: any) => {
    if (s.type !== 'ExportNamedDeclaration') return false;
    const d = s.declaration;
    if (d?.type === 'FunctionDeclaration') return d.id.name === name && directive(d);
    return d?.type === 'VariableDeclaration' && d.declarations.some((v: any) => v.id.name === name && directive(v.init));
  }) ?? false;
}
const rule: Rule.RuleModule = {
  meta: { type: 'problem', docs: { description: 'Check blaten kernel syntax and helper calls' }, schema: [], messages: {
    unsupported: 'Unsupported kernel syntax: {{syntax}}. Use const bindings, expressions and return.',
    helper: 'Kernel helper {{name}} must have a use kernel directive, or be a blaten intrinsic.',
  } },
  create(context) {
    const functions: any[] = [];
    const source = context.sourceCode;
    function variable(node: any, name: string): any {
      let scope: any = source.getScope(node);
      while (scope) { const v = scope.set.get(name); if (v) return v; scope = scope.upper }
    }
    function importedIntrinsic(node: any): Intrinsic | undefined {
      if (node?.type !== 'Identifier') return;
      const def = variable(node, node.name)?.defs[0];
      if (def?.type !== 'ImportBinding') return;
      const declaration = def.parent ?? def.node.parent;
      if (declaration.source?.value === 'blaten' && declaration.importKind !== 'type' && def.node.importKind !== 'type') {
        return intrinsics.get(def.node.imported?.name ?? def.node.imported?.value);
      }
    }
    function isHelper(node: any): boolean {
      if (node?.type !== 'Identifier') return false;
      const v = variable(node, node.name), def = v?.defs[0];
      if (!def) return false;
      if (def.type === 'ImportBinding') {
        const declaration = def.parent ?? def.node.parent;
        const from = declaration.source?.value;
        const imported = def.node.imported?.name;
        if (from === 'blaten') return !!importedIntrinsic(node);
        return typeof from === 'string' && importedKernel(context.filename, from, imported);
      }
      const fn = def.type === 'FunctionName' ? def.node : def.node.init;
      if (fn?.type === 'CallExpression' && fn.callee.type === 'Identifier') {
        const factory = variable(fn.callee, fn.callee.name)?.defs[0];
        if (factory?.type === 'ImportBinding' && factory.parent.source.value === 'blaten' && ['grad', 'valueAndGrad', 'customJvp', 'customVjp'].includes(factory.node.imported?.name)) return isHelper(fn.arguments[0]);
      }
      return directive(fn);
    }
    function inspect(node: any) {
      if (node.async || node.generator || node.params.some((p: any) => p.type !== 'Identifier')) context.report({ node, messageId: 'unsupported', data: { syntax: 'async, generator, or non-identifier parameters' } });
      const allowed = new Set(['BlockStatement', 'ExpressionStatement', 'Literal', 'Identifier', 'ReturnStatement', 'VariableDeclaration', 'VariableDeclarator', 'BinaryExpression', 'UnaryExpression', 'ArrayExpression', 'ObjectExpression', 'Property', 'CallExpression', 'MemberExpression']);
      const visit = (n: any) => {
        if (!n || typeof n !== 'object') return;
        if (n.type?.startsWith('TS')) {
          if (['TSAsExpression', 'TSNonNullExpression'].includes(n.type)) visit(n.expression);
          return;
        }
        if (n.type && !allowed.has(n.type)) { context.report({ node: n, messageId: 'unsupported', data: { syntax: n.type } }); return }
        if (n.type === 'ExpressionStatement') {
          if (n.expression?.value === 'use kernel') return;
          context.report({ node: n, messageId: 'unsupported', data: { syntax: 'expression statement' } }); return;
        }
        if (n.type === 'VariableDeclaration' && n.kind !== 'const') context.report({ node: n, messageId: 'unsupported', data: { syntax: n.kind } });
        if (n.type === 'VariableDeclarator' && (n.id.type !== 'Identifier' || !n.init)) context.report({ node: n, messageId: 'unsupported', data: { syntax: 'destructured or uninitialized binding' } });
        if (n.type === 'Property' && (n.computed || n.method || n.kind !== 'init')) context.report({ node: n, messageId: 'unsupported', data: { syntax: 'computed property, method, or accessor' } });
        if (n.type === 'BinaryExpression' && !['+', '-', '*', '/', '**', '>'].includes(n.operator)) context.report({ node: n, messageId: 'unsupported', data: { syntax: n.operator } });
        if (n.type === 'UnaryExpression' && n.operator !== '-') context.report({ node: n, messageId: 'unsupported', data: { syntax: n.operator } });
        if (n.type === 'CallExpression') {
          const method = n.callee.type === 'MemberExpression' && !n.callee.computed && tensorMethods.get(n.callee.property.name);
          if (!method && !isHelper(n.callee)) context.report({ node: n.callee, messageId: 'helper', data: { name: source.getText(n.callee) } });
          const intrinsic = method || importedIntrinsic(n.callee);
          for (const index of intrinsic?.kernelParameters ?? []) {
            const argumentIndex = method ? index - 1 : index;
            const callback = n.arguments[argumentIndex];
            if (!isHelper(callback)) context.report({ node: callback ?? n, messageId: 'helper', data: { name: callback ? source.getText(callback) : `argument ${argumentIndex + 1}` } });
          }
        }
        for (const key of source.visitorKeys[n.type] ?? []) {
          if (key === 'typeAnnotation' || key === 'returnType') continue;
          const value = n[key]; if (Array.isArray(value)) value.forEach(visit); else visit(value);
        }
      };
      visit(node.body);
    }
    return {
      FunctionDeclaration: node => { if (directive(node)) functions.push(node) },
      FunctionExpression: node => { if (directive(node)) functions.push(node) },
      ArrowFunctionExpression: node => { if (directive(node)) functions.push(node) },
      'Program:exit': () => functions.forEach(inspect),
    };
  },
};
export default { rules: { 'kernel-contract': rule } };
