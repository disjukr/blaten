import { defineConfig } from 'vite';
import solid from 'vite-plugin-solid';
import blaten from '@blaten/compiler/vite';

export default defineConfig({ plugins: [blaten({ trace: true }), solid()], build: { target: 'esnext' } });
