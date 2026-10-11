import { create, globals } from 'webgpu';
import { requestDevice as requestWebGpuDevice, type DeviceOptions, type DisposableDevice } from './device.js';

export type { DeviceOptions, DisposableDevice } from './device.js';
// Retain the native GPU provider for the lifetime of devices created from it.
let gpu: GPU | undefined;

/** Initialize Node WebGPU and request an owned device supporting `using`. */
export async function requestDevice(options: DeviceOptions = {}): Promise<DisposableDevice> {
  Object.assign(globalThis, globals);
  gpu ??= create([]);
  return requestWebGpuDevice({ ...options, gpu: options.gpu ?? gpu });
}
