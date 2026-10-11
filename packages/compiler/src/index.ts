import { transformSync } from '@babel/core';
import * as t from '@babel/types';
import type { Plugin } from 'vite';
import { posix, win32 } from 'node:path';
import { toKernelIR } from './kernel-ir.js';
import { signatureReader } from './signature.js';

export function hasKernelDirective(node: any): boolean {
  return node.body?.directives?.some((d: any) => d.value.value === 'use kernel') ?? false;
}

// Source AST and TypeScript types are consumed here; the runtime receives plain
// kernel IR and input signatures, never compiler ASTs or type objects.
export default function blaten(options: { sourceRoot?: string } = {}): Plugin {
  let sourceRoot = options.sourceRoot ?? process.cwd();
  const sourceFile = (id: string) => {
    const file = id.split('?')[0]!;
    const paths = /^[A-Za-z]:[\\/]|^\\\\/.test(file) ? win32 : posix;
    if (file.startsWith('\0')) return `virtual/${posix.basename(file.slice(1).replace(/\\/g, '/'))}`;
    if (!paths.isAbsolute(file)) return file.replace(/\\/g, '/');
    const name = paths.relative(sourceRoot, file).replace(/\\/g, '/');
    // Different Windows drives cannot be expressed relative to one another.
    return paths.isAbsolute(name) ? `external/${paths.basename(file)}` : name;
  };
  return {
    name: 'blaten:kernel', enforce: 'pre',
    configResolved(config) { sourceRoot = options.sourceRoot ?? config.root },
    transform(source, id) {
      if (id.includes('node_modules') || !/\.[cm]?[jt]sx?(?:\?|$)/.test(id) || !source.includes('use kernel')) return;
      let count = 0;
      let readSignature: ReturnType<typeof signatureReader> | undefined;
      const result = transformSync(source, {
        filename: id.split('?')[0], configFile: false, babelrc: false, sourceMaps: true,
        parserOpts: { sourceType: 'module', plugins: ['typescript'] },
        plugins: [() => ({ visitor: {
          Program: {
            enter(path: any) { path.setData('registrations', []); path.setData('register', path.scope.generateUidIdentifier('registerKernel')) },
            exit(path: any) {
              if (!count) return;
              path.node.body.unshift(t.importDeclaration([t.importSpecifier(path.getData('register'), t.identifier('registerKernel'))], t.stringLiteral('blaten/internal')), ...path.getData('registrations'));
            },
          },
          Function(path: any) {
            if (!hasKernelDirective(path.node)) return;
            if (path.node.async || path.node.generator) throw path.buildCodeFrameError('Kernels must be synchronous functions');
            if (!path.isFunctionDeclaration() && !path.isFunctionExpression() && !path.isArrowFunctionExpression()) throw path.buildCodeFrameError('Use a function declaration or function expression for kernels');
            if (path.node.params.some((p: any) => p.type !== 'Identifier')) throw path.buildCodeFrameError('Kernel parameters must be identifiers');
            const program = path.findParent((p: any) => p.isProgram());
            if (!path.parentPath.isProgram() && !path.parentPath.isExportNamedDeclaration() && !path.parentPath.isExportDefaultDeclaration() && path.isFunctionDeclaration()) throw path.buildCodeFrameError('Kernel declarations must be at module scope');
            if (path.isFunctionDeclaration() && !path.node.id) path.node.id = path.scope.generateUidIdentifier('kernel');
            const names = new Set<string>();
            path.traverse({ ReferencedIdentifier(reference: any) {
              if (reference.findParent((p: any) => p.isTSType() || p.isTSTypeAnnotation())) return;
              const binding = reference.scope.getBinding(reference.node.name);
              let scope = binding?.scope;
              while (scope && scope !== path.scope) scope = scope.parent;
              if (!scope) names.add(reference.node.name);
            } });
            let ir;
            try { ir = toKernelIR(path.node, names) }
            catch (error) { throw path.buildCodeFrameError((error as Error).message) }
            const metadata = t.objectExpression([
              t.objectProperty(t.identifier('version'), t.numericLiteral(2)),
              t.objectProperty(t.identifier('name'), t.stringLiteral(path.node.id?.name ?? path.parentPath.node.id?.name ?? 'kernel')),
              t.objectProperty(t.identifier('ir'), t.valueToNode(ir)),
              ...Object.entries((readSignature ??= signatureReader(source, id))(path.node.start)).map(([key, value]) => t.objectProperty(t.identifier(key), t.valueToNode(value))),
              t.objectProperty(t.identifier('externals'), t.objectExpression([...names].map(name => t.objectProperty(t.identifier(name), t.arrowFunctionExpression([], t.identifier(name)))))),
              t.objectProperty(t.identifier('source'), t.valueToNode({ file: sourceFile(id), start: path.node.start, end: path.node.end })),
            ]);
            count++;
            if (path.isFunctionDeclaration()) program.getData('registrations').push(t.expressionStatement(t.callExpression(program.getData('register'), [t.identifier(path.node.id.name), metadata])));
            else {
              const original = path.node;
              path.replaceWith(t.callExpression(program.getData('register'), [original, metadata]));
              path.skip();
            }
          },
        } })],
      });
      return result?.code ? { code: result.code, map: result.map } : undefined;
    },
  };
}
