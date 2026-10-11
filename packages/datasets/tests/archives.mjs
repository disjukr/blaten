import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { c as createTar } from 'tar';
import { downloadFile, prepareArchive, extractArchive, inside } from '../dist/files.js';
const root = await mkdtemp(join(tmpdir(), 'blaten-archives-'));
const originalFetch = globalThis.fetch;
function crc32(bytes) { let n = 0xffffffff; for (const b of bytes) { n ^= b; for (let i = 0; i < 8; i++) n = n >>> 1 ^ (n & 1 ? 0xedb88320 : 0) } return (n ^ 0xffffffff) >>> 0 }
function zip(name, value) {
  const filename = Buffer.from(name), data = Buffer.from(value), local = Buffer.alloc(30), central = Buffer.alloc(46), end = Buffer.alloc(22), crc = crc32(data);
  local.writeUInt32LE(0x04034b50); local.writeUInt16LE(20, 4); local.writeUInt32LE(crc, 14); local.writeUInt32LE(data.length, 18); local.writeUInt32LE(data.length, 22); local.writeUInt16LE(filename.length, 26);
  central.writeUInt32LE(0x02014b50); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt32LE(crc, 16); central.writeUInt32LE(data.length, 20); central.writeUInt32LE(data.length, 24); central.writeUInt16LE(filename.length, 28);
  const first = Buffer.concat([local, filename, data]), directory = Buffer.concat([central, filename]); end.writeUInt32LE(0x06054b50); end.writeUInt16LE(1, 8); end.writeUInt16LE(1, 10); end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(first.length, 16);
  return Buffer.concat([first, directory, end]);
}
try {
  assert.equal(inside(root, '..allowed'), join(root, '..allowed')); assert.throws(() => inside(root, '../escape'), /Invalid/);
  await mkdir(join(root, 'source/data'), { recursive: true }); await writeFile(join(root, 'source/data/a.txt'), 'fixture');
  await createTar({ file: join(root, 'archive.tgz'), cwd: join(root, 'source'), gzip: true }, ['data']);
  const tar = await readFile(join(root, 'archive.tgz')), md5 = createHash('md5').update(tar).digest('hex'); let requests = 0;
  globalThis.fetch = async () => { requests++; return new Response(tar) };
  const file = { url: 'https://provider.invalid/archive.tgz', md5 };
  await Promise.all([prepareArchive(join(root, 'cache'), file, 'data/a.txt'), prepareArchive(join(root, 'cache'), file, 'data/a.txt')]); assert.equal(requests, 1);
  assert.equal(await readFile(join(root, 'cache/data/a.txt'), 'utf8'), 'fixture');
  await rm(join(root, 'cache/archive.tgz.extracted')); await writeFile(join(root, 'cache/data/a.txt'), 'interrupted');
  await prepareArchive(join(root, 'cache'), file, 'data', false); assert.equal(await readFile(join(root, 'cache/data/a.txt'), 'utf8'), 'fixture', 'An interrupted extraction is retried even when a directory exists');
  await prepareArchive(join(root, 'cache'), file, 'data', false); assert.equal(requests, 1);
  await assert.rejects(downloadFile(root, { url: 'https://provider.invalid/missing' }, false), /missing/); assert.equal(requests, 1);
  await writeFile(join(root, 'good.zip'), zip('data/z.txt', 'zip')); await extractArchive(join(root, 'good.zip'), join(root, 'zip')); assert.equal(await readFile(join(root, 'zip/data/z.txt'), 'utf8'), 'zip');
  await writeFile(join(root, 'bad.zip'), zip('../outside.txt', 'bad')); await assert.rejects(extractArchive(join(root, 'bad.zip'), join(root, 'zip')), /Invalid dataset path/);
  await mkdir(join(root, 'outside')); await symlink(join(root, 'outside'), join(root, 'zip/link')); await writeFile(join(root, 'link.zip'), zip('link/escape.txt', 'bad')); await assert.rejects(extractArchive(join(root, 'link.zip'), join(root, 'zip')), /symlink/); assert.deepEqual(await readdir(join(root, 'outside')), []);
  console.log('Passed: streamed downloads, offline archives, shared extraction, interrupted extraction recovery, ZIP/TAR decoding, and archive path checks.');
} finally { globalThis.fetch = originalFetch; await rm(root, { recursive: true, force: true }) }
