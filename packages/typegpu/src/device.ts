export interface DeviceOptions extends GPUDeviceDescriptor {
  gpu?: GPU;
  adapterOptions?: GPURequestAdapterOptions;
}
export type DisposableDevice = GPUDevice & Disposable;

/** Request an owned device. Use `using`, or destroy it explicitly. */
export async function requestDevice(options: DeviceOptions = {}): Promise<DisposableDevice> {
  const { gpu = globalThis.navigator?.gpu, adapterOptions, ...descriptor } = options;
  if (!gpu) throw new Error('WebGPU is unavailable. In Node, import requestDevice from @blaten/typegpu/node.');
  const adapter = await gpu.requestAdapter(adapterOptions);
  if (!adapter) throw new Error('No WebGPU adapter found');
  const device = await adapter.requestDevice({ ...descriptor,
    requiredFeatures: descriptor.requiredFeatures ?? (adapter.features.has('shader-f16') ? ['shader-f16'] : []),
  });
  if (!(Symbol.dispose in device)) Object.defineProperty(device, Symbol.dispose, { value: () => device.destroy() });
  return device as DisposableDevice;
}
