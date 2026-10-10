import ts from 'typescript';
import { resolve, dirname } from 'node:path';
import type { Signature, SignatureConstraints, ValueConstraint, DType } from 'blaten';

const dtypes = new Set(['bool', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'f16', 'f32', 'f64']);

/** Resolve imported aliases and tuple types during the build; retain only plain data. */
export function signatureReader(source: string, id: string): (start: number) => { signature?: Signature; signatureError?: string; constraints?: SignatureConstraints } {
  const filename = resolve(id.split('?')[0]!);
  const configPath = ts.findConfigFile(dirname(filename), ts.sys.fileExists);
  const config = configPath ? ts.readConfigFile(configPath, ts.sys.readFile).config : {};
  // Only options are needed: do not enumerate the config's source tree (or a
  // whole drive for virtual modules outside a project).
  const options: ts.CompilerOptions = {
    target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler,
    ...ts.parseJsonConfigFileContent({ ...config, files: [], include: [] }, ts.sys, configPath ? dirname(configPath) : dirname(filename)).options,
  };
  const host = ts.createCompilerHost(options);
  const getSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (path, languageVersion, onError, shouldCreateNewSourceFile) =>
    resolve(path) === filename ? ts.createSourceFile(path, source, languageVersion, true) : getSourceFile(path, languageVersion, onError, shouldCreateNewSourceFile);
  const program = ts.createProgram([filename], { ...options, noEmit: true }, host);
  const checker = program.getTypeChecker();
  const file = program.getSourceFile(filename)!;
  const functions = new Map<number, ts.SignatureDeclaration>();
  const visit = (node: ts.Node) => {
    if (ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isArrowFunction(node)) {
      // Babel's function start excludes export modifiers.
      const token = ts.isFunctionDeclaration(node) ? node.getChildren(file).find(child => child.kind === ts.SyntaxKind.FunctionKeyword) : undefined;
      functions.set(token?.getStart(file) ?? node.getStart(file), node);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  function valueType(type: ts.Type, at: ts.Node): ValueConstraint {
    const shapeProperty = type.getProperty('shape'), dtypeProperty = type.getProperty('dtype');
    if (shapeProperty && dtypeProperty) {
      const shapeType = checker.getTypeOfSymbolAtLocation(shapeProperty, at);
      let dtypeType = checker.getTypeOfSymbolAtLocation(dtypeProperty, at);
      dtypeType = checker.getBaseConstraintOfType(dtypeType) ?? dtypeType;
      const dtypeParts = dtypeType.isUnion() ? dtypeType.types : [dtypeType];
      const allowed = dtypeParts.map(part => {
        if (!(part.flags & ts.TypeFlags.StringLiteral) || !dtypes.has((part as ts.StringLiteralType).value)) throw new Error('Tensor dtype must constrain supported dtypes');
        return (part as ts.StringLiteralType).value as DType;
      });
      let shape: (number | null)[] | null = null;
      if (checker.isTupleType(shapeType)) {
        const dimensions = checker.getTypeArguments(shapeType as ts.TypeReference);
        const target = (shapeType as ts.TupleTypeReference).target;
        if (target.elementFlags.some(flag => flag !== ts.ElementFlags.Required)) throw new Error('Tensor shape cannot have optional or rest dimensions');
        shape = dimensions.map(dim => {
          if (dim.flags & ts.TypeFlags.NumberLiteral) {
            const value = (dim as ts.NumberLiteralType).value;
            if (!Number.isSafeInteger(value) || value <= 0) throw new Error('Tensor shape dimensions must be positive integer literals');
            return value;
          }
          if (dim.flags & ts.TypeFlags.Number || checker.getBaseConstraintOfType(dim)?.flags === ts.TypeFlags.Number) return null;
          throw new Error('Unsupported tensor dimension type');
        });
      }
      return { kind: 'tensor', shape, dtypes: allowed };
    }
    if (type.flags & ts.TypeFlags.NumberLike) return 'f32';
    if (type.flags & ts.TypeFlags.BigIntLike) return 'i64';
    if (type.flags & ts.TypeFlags.BooleanLike || type.isUnion() && type.types.every(part => !!(part.flags & ts.TypeFlags.BooleanLike))) return 'bool';
    throw new Error(`Unsupported input type ${checker.typeToString(type)}`);
  }
  return start => {
    try {
      const fn = functions.get(start);
      if (!fn) throw new Error('Function type information is unavailable');
      const inputs = fn.parameters.map(parameter => {
        if (parameter.questionToken || parameter.dotDotDotToken || parameter.initializer) throw new Error('Kernel inputs cannot be optional, rest, or default parameters');
        return valueType(checker.getTypeAtLocation(parameter), parameter);
      });
      const constraints = { inputs };
      if (inputs.some(input => typeof input !== 'string' && (input.shape === null || input.shape.includes(null) || input.dtypes.length !== 1))) {
        return { constraints, signatureError: 'Input shape or dtype is not concrete; supply compile options.inputs' };
      }
      return { constraints, signature: { inputs: inputs.map(input => typeof input === 'string' ? input : ({ kind: 'tensor', shape: input.shape as number[], dtype: input.dtypes[0]! })) } };
    } catch (error) {
      return { signatureError: (error as Error).message };
    }
  };
}
