import { create, globals } from 'webgpu';
import { typegpu, type TypeGpuBackend } from '@blaten/typegpu';

export async function withBackend<T>(run: (backend: TypeGpuBackend, adapter: GPUAdapter) => Promise<T>): Promise<T> {
  Object.assign(globalThis, globals);
  const gpu = create([]), adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available');
  const device = await adapter.requestDevice(), errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  let backend: TypeGpuBackend | undefined;
  try {
    backend = await typegpu({ device });
    const result = await run(backend, adapter);
    await backend.synchronize();
    if (errors.length) throw new Error(errors.join('\n'));
    return result;
  } finally { backend?.dispose(); device.destroy() }
}
