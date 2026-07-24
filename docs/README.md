# Documentation Workflow

The webpage is built directly from Markdown files in `docs/src`.

## Edit Pages

- UniPolfed landing page: `docs/src/index.md`
- Getting started: `docs/src/getting-started/index.md`
- Beginner tutorials: `docs/src/tutorials/beginner/<page-name>/index.md`
- Advanced tutorials: `docs/src/tutorials/advanced/<page-name>/index.md`
- Models: `docs/src/models/<model-name>/index.md`
- API/reference pages: `docs/src/documentation/<page-name>/index.md`
- FAQ: `docs/src/faq/index.md`

The URL follows the same folder structure. For example:

- `docs/src/index.md` becomes `/`
- `docs/src/tutorials/beginner/optimized-mapping/index.md` becomes `/tutorials/beginner/optimized-mapping/`
- `docs/src/models/qsun/index.md` becomes `/models/qsun/`
- `docs/src/documentation/models/index.md` becomes `/documentation/models/`

## Sidebar

The sidebar and page order are controlled in `docs/make.jl`.

When adding a new page:

1. Create a Markdown file under `docs/src/.../index.md`.
2. Add it to the `pages = [...]` list in `docs/make.jl`.
3. Run `bash build.sh` from `docs/`.

## Build

From the docs folder:

```bash
bash build.sh
```

To select a specific Julia executable, use:

```bash
JULIA_BIN=/path/to/julia bash build.sh
```

The local output is written to `docs/build`.

## Publish

Publishing is handled exclusively by the GitHub Pages workflow; the local build
script never writes outside this repository.

The docs are published under:

```text
https://rockclimbingrocks.github.io/UniPolfed.jl/
```

Subpages follow the same path structure:

```text
https://rockclimbingrocks.github.io/UniPolfed.jl/citation/
https://rockclimbingrocks.github.io/UniPolfed.jl/getting-started/
```

The landing page content is `docs/src/index.md`.
