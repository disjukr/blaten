import tseslint from 'typescript-eslint';
import blaten from '@blaten/eslint-plugin';
export default [
  { ignores: ['**/dist/**', '**/node_modules/**', '**/.cache/**'] },
  { files: ['**/*.{ts,tsx}'], languageOptions: { parser: tseslint.parser }, plugins: { blaten }, rules: { 'blaten/kernel-contract': 'error' } },
];
