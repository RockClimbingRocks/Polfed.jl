# Documentation Workflow

The webpage is built directly from Markdown files in `docs/src`.

## Edit Pages

- Home page: `docs/src/index.md`
- Getting started: `docs/src/getting-started/index.md`
- Tutorials: `docs/src/tutorials/<page-name>/index.md`
- Models: `docs/src/models/<model-name>/index.md`
- API/reference pages: `docs/src/documentation/<page-name>/index.md`
- FAQ: `docs/src/faq/index.md`

The URL follows the same folder structure. For example:

- `docs/src/tutorials/choosing-target/index.md` becomes `/tutorials/choosing-target/`
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

On machines where plain `julia` points to an old Julia version, use:

```bash
JULIA_BIN=/Users/rockclimbingrocks/.juliaup/bin/julia bash build.sh
```

The local output is written to `docs/build`.

## Publish

GitHub Pages publishing is handled by Documenter through `docs/make.jl`.

The current development docs are published under:

```text
https://rockclimbingrocks.github.io/Polfed.jl/dev/
```

The public root page is `docs/src/index.md`.
