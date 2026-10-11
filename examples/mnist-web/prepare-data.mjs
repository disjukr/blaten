import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';

// Serve official files with the application: no runtime CORS dependency.
const files = [
  ['train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'],
  ['train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'],
  ['t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'],
  ['t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c'],
];
const root = new URL('./public/mnist/', import.meta.url);
await mkdir(root, { recursive: true });
for (const [name, md5] of files) {
  const valid = value => createHash('md5').update(value).digest('hex') === md5;
  const target = new URL(name, root);
  const existing = await readFile(target).catch(error => { if (error.code === 'ENOENT') return; throw error; });
  if (existing && valid(existing)) continue;
  let bytes = await readFile(new URL(`../mnist/data/mnist/${name}`, import.meta.url)).catch(error => { if (error.code === 'ENOENT') return; throw error; });
  if (!bytes || !valid(bytes)) {
    console.log(`Downloading official MNIST: ${name}`);
    const response = await fetch(`https://storage.googleapis.com/cvdf-datasets/mnist/${name}`);
    if (!response.ok) throw new Error(`MNIST download failed: ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
  }
  if (!valid(bytes)) throw new Error(`MNIST checksum mismatch: ${name}`);
  await writeFile(target, bytes);
}
