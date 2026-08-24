# Documentation Workflow

The webpage is built directly from Markdown files in `docs/src`.

The previous version of the site is kept, unbuilt, in `docs-old/` for reference. Nothing
there is published — only `docs/` is built and deployed. See `SITE-PLAN.md` for the
structure decisions behind the current layout.

## Site structure

```text
Home                    docs/src/index.md
Getting Started         getting-started/{installation,quickstart}/
How POLFED works        how-polfed-works/{,krylov,filters,on-the-fly-mapping}/
Tutorials               tutorials/{polfed,bring-your-own-operator,patched-polfed,
                                   full-polfed,double-polfed,unitary-polfed}/
Going faster            going-faster/{,optimized-kernels,parallelization,gpu}/
Models                  models/{qsun,xxz,j1j2}/
Guidelines              guidelines/
Benchmarks              benchmarks/
API Reference           api/{entrypoints,configs-and-targets,
                             results-reports-logging,models}/
FAQ / Troubleshooting   faq/
Development             development/{architecture,roadmap}.md
Citation                citation/
```

Ordering rationale: Getting Started comes before How POLFED works, so a first visitor
reaches `] add` before the theory. The 30-second teaser lives on Home only.

## Edit Pages

The URL follows the folder structure. For example:

- `docs/src/index.md` becomes `/`
- `docs/src/getting-started/quickstart/index.md` becomes `/getting-started/quickstart/`
- `docs/src/going-faster/gpu/index.md` becomes `/going-faster/gpu/`
- `docs/src/api/entrypoints/index.md` becomes `/api/entrypoints/`

## Conventions

- **Every tutorial ends with a "Reading the output" section.** `PolfedResult` versus
  `PatchedReport` is the most common point of confusion, so each tutorial explains the
  output it actually produces rather than deferring to one shared page.
- **Model pages open with a "you do not need this model" note.** The shipped Hamiltonians
  are conveniences; the risk is readers thinking POLFED only works on them.
- **The API Reference excludes internals.** Only what a user can see and call.
- **Cross-reference anchors** use the `UniPolfed.`-prefixed form for API objects
  (`@ref UniPolfed.polfed`) and short slugs for pages (`@ref tutorial_polfed`).
- **Diagrams are inline SVG**, theme-aware via `html.theme--documenter-dark` and the
  catppuccin variants. Mermaid would be more diffable but needs `DocumenterMermaid` added
  to `docs/Project.toml`; that swap has not been made.
- **Placeholder pages** (Double POLFED, Unitary POLFED, Benchmarks) carry an explicit
  admonition saying so, plus a checklist of what to write.

## Sidebar

The sidebar and page order are controlled in `docs/make.jl`.

When adding a new page:

1. Create a Markdown file under `docs/src/.../index.md`.
2. Add it to the `pages = [...]` list in `docs/make.jl`.
3. Run `bash build.sh` from `docs/`.

`pagesonly=true` is set, so a file that is not listed in `make.jl` is not built.

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

Publishing is handled exclusively by the GitHub Pages workflow; the local build script
never writes outside this repository.

The docs are published under:

```text
https://rockclimbingrocks.github.io/UniPolfed.jl/
```

Subpages follow the same path structure:

```text
https://rockclimbingrocks.github.io/UniPolfed.jl/citation/
https://rockclimbingrocks.github.io/UniPolfed.jl/getting-started/quickstart/
```

The landing page content is `docs/src/index.md`.
