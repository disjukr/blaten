/** Portable inspection data, independent of executables and device resources. */
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export interface CompileInfo {
  name: string;
  backend: string;
  cause: 'explicit' | 'direct';
}
export interface Diagnostic { name: string; message: string; stack?: string }
/** Artifact references resolve in the compilation's artifacts or TraceResult.sources. */
export interface Stage { id: string; name: string; inputs: string[]; artifacts: string[] }
export interface Artifact {
  id: string;
  kind: string;
  schemaVersion: number;
  location?: { file: string; start: number; end: number };
  content: { type: 'json'; value: JsonValue } | { type: 'text'; language: string; value: string }
    | { type: 'binary'; mediaType: string; encoding: 'base64'; value: string };
}
export interface Compilation extends CompileInfo {
  id: string;
  status: 'pending' | 'success' | 'failed';
  stages: Stage[];
  artifacts: Artifact[];
  diagnostics: Diagnostic[];
}
export interface TraceResult {
  format: 'blaten-trace';
  version: 1;
  /** Full original files, shared across compilations. IDs are trace-wide. */
  sources: Artifact[];
  compilations: Compilation[];
}
export interface Recorder {
  readonly id: string;
  addStage(stage: Stage): void;
  addArtifact(artifact: Artifact): void;
  complete(): void;
  fail(error: Diagnostic): void;
}
export interface Trace {
  begin(info: CompileInfo): Recorder;
  /** A detached copy of all records so far; recording continues. */
  toResult(): TraceResult;
}

function clone<T>(value: T): T { return JSON.parse(JSON.stringify(value)) as T }

/** Attach full original files once and link kernel ranges to them. Missing files are left unresolved. */
export function attachSources(result: TraceResult, sources: Readonly<Record<string, string>>): void {
  for (const [file, value] of Object.entries(sources)) {
    const existing = result.sources.findIndex(a => a.location?.file === file);
    const id = existing < 0 ? `s${result.sources.length + 1}` : result.sources[existing]!.id;
    const source: Artifact = { id, kind: 'source', schemaVersion: 1, location: { file, start: 0, end: value.length },
      content: { type: 'text', language: 'typescript', value } };
    if (existing < 0) result.sources.push(source);
    else result.sources[existing] = source;
  }
  for (const compilation of result.compilations) {
    const artifacts: string[] = [];
    for (const artifact of compilation.artifacts.filter(a => a.kind === 'blaten.kernel-ir')) {
      if (artifact.content.type !== 'json') continue;
      const { kernels } = artifact.content.value as unknown as { kernels: { id: string; kind: string; source?: { file: string; start: number; end: number; artifact?: string } }[] };
      for (const kernel of kernels) {
        if (kernel.kind !== 'kernel' || !kernel.source) continue;
        const source = result.sources.find(a => a.location?.file === kernel.source!.file);
        if (!source) continue;
        kernel.source.artifact = source.id;
        artifacts.push(source.id);
      }
    }
    if (!artifacts.length) continue;
    const stage = compilation.stages.find(s => s.id === 'source');
    if (stage) stage.artifacts = [...new Set([...stage.artifacts, ...artifacts])];
    else compilation.stages.unshift({ id: 'source', name: 'TypeScript source', inputs: [], artifacts: [...new Set(artifacts)] });
    const kernelStage = compilation.stages.find(s => s.id === 'kernel');
    if (kernelStage && !kernelStage.inputs.includes('source')) kernelStage.inputs.push('source');
  }
}

export function createTrace(): Trace {
  const compilations: Compilation[] = [];
  return {
    begin(info) {
      const record: Compilation = { ...clone(info), id: `c${compilations.length + 1}`, status: 'pending', stages: [], artifacts: [], diagnostics: [] };
      compilations.push(record);
      return {
        id: record.id,
        addStage(stage) { record.stages.push(clone(stage)) },
        addArtifact(artifact) { record.artifacts.push(clone(artifact)) },
        complete() { record.status = 'success' },
        fail(error) { record.diagnostics.push(clone(error)); record.status = 'failed' },
      };
    },
    toResult() { return clone({ format: 'blaten-trace', version: 1, sources: [], compilations }) },
  };
}
