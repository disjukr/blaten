interface Split { pixels: Uint8Array; labels: Uint8Array }
const origin = `${import.meta.env.BASE_URL}mnist/`;
const loaded = new Map<'train' | 'test', Split>();

async function download(name: string, signal: AbortSignal) {
  const response = await fetch(origin + name + '.gz', { signal, cache: 'force-cache' });
  if (!response.ok || !response.body) throw new Error(`MNIST download failed: ${response.status} (${name})`);
  const buffer = await response.arrayBuffer(), bytes = new Uint8Array(buffer);
  // Some static servers mark .gz as Content-Encoding:gzip; fetch already decodes it.
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  return new Uint8Array(await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}

export async function loadSplit(split: 'train' | 'test', signal: AbortSignal): Promise<Split> {
  signal.throwIfAborted();
  const cached = loaded.get(split);
  if (cached) return cached;
  const prefix = split === 'train' ? 'train' : 't10k';
  const [images, labels] = await Promise.all([download(`${prefix}-images-idx3-ubyte`, signal), download(`${prefix}-labels-idx1-ubyte`, signal)]);
  if (images.length < 16 || labels.length < 8) throw new Error('Invalid MNIST IDX headers.');
  const iv = new DataView(images.buffer), lv = new DataView(labels.buffer), count = iv.getUint32(4);
  if (iv.getUint32(0) !== 2051 || iv.getUint32(8) !== 28 || iv.getUint32(12) !== 28 || lv.getUint32(0) !== 2049 || lv.getUint32(4) !== count || count !== (split === 'train' ? 60000 : 10000) || images.length !== 16 + count * 784 || labels.length !== 8 + count || labels.subarray(8).some(n => n > 9)) throw new Error('Invalid MNIST IDX data.');
  const result = { pixels: images.subarray(16), labels: labels.subarray(8) };
  loaded.set(split, result);
  return result;
}
