import { fileURLToPath } from 'node:url';
import { createServer, createServerModuleRunner } from 'vite';
import blaten from '@blaten/compiler/vite';
const server = await createServer({
  root: fileURLToPath(new URL('.', import.meta.url)), configFile: false, plugins: [blaten()],
  server: { middlewareMode: true, hmr: false, watch: null }, appType: 'custom',
  ssr: { external: ['blaten', '@blaten/typegpu', 'typegpu', 'webgpu'] },
});
const runner = createServerModuleRunner(server.environments.ssr, { hmr: false });
try { await (await runner.import('/src/run.ts')).main() }
finally { await runner.close(); await server.close() }
