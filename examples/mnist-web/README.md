# MNIST in the browser

A Solid + Vite example using Blaten's TypeGPU backend. The model is defined
locally in `src/model.ts` and displayed at the top of the page before compilation.

From the repository root:

```sh
pnpm install
pnpm build
pnpm --filter @blaten/mnist-web dev
```

Open the printed localhost URL in a WebGPU-capable Chrome or Edge. Production
deployments need HTTPS. This example requires WebGPU; there is no CPU fallback.

- **Start training** trains the seeded 784 → 32 → 10 model on MNIST,
  and measures held-out test accuracy after every epoch. Settings select the
  first N training/test images. Training shuffles and drops the final incomplete
  batch; evaluation pads it and counts only valid images.
- Opening the page prepares training/inference and shows initial weights and
  trace without executing either model kernel. The dataset preview separately
  loads training images and shares them with training.
  The device, backend, and lazy compilation caches stay alive for the page's
  lifetime. Starting training continues from current weights and reuses the prepared model
  kernels. New batch shapes compile once and remain cached. Optimizer and metric
  shaders compile on first use and are cached by the backend.
- The model source uses TypeScript syntax highlighting. Click a `'use kernel'`
  function (or focus it and press Enter) to filter compilation results and show
  a compilation. Multiple candidates open a context menu with recorded
  call/derivative paths. `predict` has its own compilation and appears
  through `loss` in `valueAndGrad(loss)`. Paths start at the compilation entry:
  `valueAndGrad(loss)` → `loss` → `predict`. Tensor IR
  and WGSL describe the selected whole compilation, not a per-function node
  subset. Tensor nodes link to their Kernel IR instructions where direct
  correspondence is known; generated derivatives can have no direct origin.
- **Stop** aborts data requests or stops between GPU batches. Runs accumulate
  trained weights and epoch history. Check **Reset weights** to restore seeded
  initial weights and clear history before a run. Run-owned tensors are disposed
  after each run; model weights, the device, backend,
  and cached executables are disposed when the page component is removed.
- Weight tiles show each hidden neuron's 28 × 28 input weights and bias. The
  output matrix shows rows of hidden neurons and columns of digit classes.
  Current/initial views share a symmetric color scale; change shows the
  difference from initialization. Weights update periodically and per epoch.
- The training dataset uses a virtualized, scrollable image grid with digit
  filtering. Only visible rows and a small buffer are rendered. Selecting an
  image loads its original 28 × 28 pixels into **Test the model**.
- **Test the model** accepts dataset images or a digit drawn with a mouse, pen, or touch. The
  drawing is resized to a 20 × 20 bounding box and centered in a 28 × 28 input.
  Predictions update automatically while drawing, using the current learned
  weights and cached inference kernel. Requests are throttled with one GPU
  inference in flight; results from cleared or superseded input are discarded.
  The result shows the predicted digit and softmax scores for digits 0–9.
  Testing does not update weights or training history. **Clear** resets the pad.
- The trace viewer accepts only a `TraceResult` object. Explore Kernel IR,
  Tensor IR, and the combined execution plan / WGSL tab. Full model source
  is explicitly attached once, with kernel ranges.
  Both IR tabs show complete directed graphs with selectable nodes, highlighted
  dependencies, zoom, and a scrollable canvas. Kernel IR follows the function
  selected in the model source and includes all its instructions; Tensor IR shows
  the compiled program's operations, shapes, and dtypes. Arrows run from values
  to consumers. Compact graph nodes show IDs and operations; instruction details
  are available in the selected node panel.
  The execution plan uses the same graph controls to connect input buffers and
  dispatches. Selecting a dispatch shows its exact WGSL and resource details.
  Node details open beside the graph, or below it on narrow screens. Close the
  details to clear node selection. The compilation section is always visible,
  starts on TS, and opens Kernel IR after choosing a compilation. Tensor nodes expose
  their JSON in the detail panel.
  Kernel nodes list their original TS line/column locations with highlighted
  excerpts above the node JSON. Mappings on the same line share a snippet with
  multiple highlights. The TS tab highlights the selected node's ranges;
  reused values list all occurrences. The compiler plugin enables `trace: true`
  to provide instruction spans, without embedding source text in metadata.
  Trace describes compilation, not GPU timings or every runtime invocation.
  WGSL details link to the dispatch's Tensor IR output, fused operations, and
  resources. Code blocks scroll horizontally for long lines.

`dev` and `build` prepare the four official MNIST gzip files under
`public/mnist/`, reusing the CLI example's cache if present or downloading from
Google's public MNIST storage. Each file is checked against the published MD5
used by `@blaten/datasets`. The generated files are ignored by Git. The browser
fetches from the application's own origin, avoiding third-party CORS issues.
The build output includes the data (~12 MB), so `dist/` is a self-contained
static site. After the first download, cached files support offline builds.

```sh
pnpm --filter @blaten/mnist-web typecheck
pnpm --filter @blaten/mnist-web build
pnpm --filter @blaten/mnist-web preview
```

The Vite build targets modern browsers supporting WebGPU and explicit resource
management. For deployment under a path prefix, set Vite's `base`; dataset URLs
use `import.meta.env.BASE_URL`.
