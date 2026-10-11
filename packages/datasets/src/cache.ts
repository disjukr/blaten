import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile, rename, rm } from 'node:fs/promises';
import { dirname } from 'node:path';
import { gunzipSync } from 'node:zlib';

interface FileSource { readonly url: string; readonly path: string; readonly md5: string; readonly download?: boolean }
const pending = new Map<string, Promise<Buffer>>();
const matches = (data: Buffer, md5: string) => createHash('md5').update(data).digest('hex') === md5;

async function retrieve({ url, path, md5, download = true }: FileSource): Promise<Buffer> {
  let cached: Buffer | undefined;
  try { cached = await readFile(path) }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
  if (cached && matches(cached, md5)) return gunzipSync(cached);
  if (!download) throw new Error(`Dataset file ${cached ? 'has an invalid checksum' : 'is missing'}: ${path}. Enable download to retrieve it.`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Dataset download failed (${response.status}): ${url}`);
  const compressed = Buffer.from(await response.arrayBuffer());
  if (!matches(compressed, md5)) throw new Error(`Dataset checksum mismatch: ${url}`);
  const decoded = gunzipSync(compressed);
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.tmp`;
  try { await writeFile(temporary, compressed, { flag: 'wx' }); await rename(temporary, path) }
  finally { await rm(temporary, { force: true }) }
  return decoded;
}

/** Coalesce concurrent requests; validate before atomically publishing a cache file. */
export async function cachedFile(source: FileSource): Promise<Buffer> {
  const key = JSON.stringify([source.path, source.md5, source.download ?? true]);
  const existing = pending.get(key);
  if (existing) return existing;
  const request = retrieve(source); pending.set(key, request);
  try { return await request } finally { pending.delete(key) }
}
