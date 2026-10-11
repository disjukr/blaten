/** Cleanup state must not reference the object whose lifetime it manages. */
export class Resource implements Disposable {
  private disposed = false;
  constructor(private cleanup: (() => void) | undefined) {}
  get isDisposed() { return this.disposed }
  assertAlive() { if (this.disposed) throw new Error('Resource is disposed') }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const cleanup = this.cleanup;
    this.cleanup = undefined;
    cleanup?.();
  }
  [Symbol.dispose](): void { this.dispose() }
}

const finalizers = new FinalizationRegistry<Resource>(resource => resource.dispose());

export function registerResource(target: object, resource: Resource): void {
  finalizers.register(target, resource, target);
}
export function disposeResource(target: object, resource: Resource): void {
  finalizers.unregister(target);
  resource.dispose();
}

export function managed<T extends object>(target: T, resource: Resource): T & Disposable & { dispose(): void } {
  registerResource(target, resource);
  const dispose = () => disposeResource(target, resource);
  return Object.assign(target, { dispose, [Symbol.dispose]: dispose });
}
