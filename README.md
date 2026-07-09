# AI Token Saver

AI Token Saver is a local CLI that helps developers spend fewer AI tokens before
they paste code, logs, or project context into an LLM.

Version 0.3 adds an evidence-first packing engine. Instead of blindly packing
whole files and trimming from the middle, it splits files into structural chunks,
scores them by evidence value, removes near-duplicates, adds compact code
skeletons, fills the token budget with the highest-value context, and can render
that context into dense PNG pages for multimodal LLMs.

It does four practical things:

- Finds the files and logs that consume the most context.
- Removes common waste such as repeated lines and terminal control noise.
- Extractively squeezes long text against a task/query.
- Builds an evidence-ranked markdown context pack that fits a token budget.
- Reports loss proxy metrics so savings do not hide missing evidence.
- Optionally renders a screenshot pack for image-capable models.

No API key is required. The default counter is heuristic, and you can install the
optional `tiktoken` extra when you want model-aware estimates.

## Install

```bash
python -m pip install -e .
```

Optional model-aware counting:

```bash
python -m pip install -e ".[accurate]"
```

Optional screenshot pack rendering:

```bash
python -m pip install -e ".[image]"
```

## Quick Start

Audit the current project:

```bash
ai-token-saver audit .
```

Compact a noisy log:

```bash
ai-token-saver trim examples/noisy-log.txt -o cleaned-log.txt
```

Extract the most relevant parts of a long document:

```bash
ai-token-saver trim README.md \
  --strategy extractive \
  --budget 700 \
  --query "token savings context compression"
```

Create an LLM-ready context pack under 8,000 tokens:

```bash
ai-token-saver pack . \
  --budget 8000 \
  --query "explain the token compression architecture" \
  -o context-pack.md
```

Use aggressive mode when cost matters more than completeness:

```bash
ai-token-saver pack . \
  --budget 3000 \
  --query "token compression evidence budget code context" \
  --mode aggressive \
  -o context-pack.md
```

Measure loss proxies for the same pack:

```bash
ai-token-saver measure . \
  --budget 3000 \
  --query "token compression evidence budget code context" \
  --mode aggressive
```

Render the selected context as dense PNG pages:

```bash
ai-token-saver shotpack . \
  --budget 3000 \
  --query "token compression evidence budget code context" \
  --mode aggressive \
  -o shotpack
```

Use a custom token price to estimate input cost:

```bash
ai-token-saver audit . --price-per-million 0.50
```

## Why This Exists

AI tools are easy to overfeed. Large logs, repeated stack traces, generated
folders, and unrelated source files can burn context before the model reaches
the important part. AI Token Saver gives you a deterministic preflight step:
measure first, trim obvious waste, then send a smaller context pack.

## How The Engine Saves More

The upgraded packer is based on a practical lesson from prompt compression and
RAG tooling: selection usually beats blind compression.

1. **Structure-aware splitting**: Markdown headings, code symbols, paragraphs,
   and file boundaries become independent chunks.
2. **Repo-map skeletons**: Code files get compact skeleton chunks that preserve
   imports, classes, functions, interfaces, and constants without full bodies.
3. **Evidence scoring**: Chunks are ranked by query overlap, path relevance,
   structural value, information density, and position.
4. **Near-duplicate filtering**: Repeated or semantically similar chunks are
   skipped before they enter the pack.
5. **Budget-aware packing**: The best score-per-token chunks are selected first,
   then the final markdown is tightened to the requested budget.
6. **Optical screenshot pack**: The selected context can be rendered into dense
   PNG pages. This is useful when a multimodal model charges fewer image tokens
   than equivalent text tokens.

This is inspired by public work in the ecosystem:

- [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua): prompt
  compression can remove non-essential prompt tokens and reach high compression
  ratios.
- [mo-tunn/TokenPack](https://github.com/mo-tunn/TokenPack): selecting useful
  evidence before compression is often stronger than compressing everything.
- [yamadashy/repomix](https://github.com/yamadashy/repomix): repository packing
  needs token counting, ignore rules, and AI-friendly output formats.
- [aider-ai/aider](https://github.com/Aider-AI/aider): codebase maps help LLMs
  work with large repos without loading every full source file.
- [Sean Goedecke's text-as-image analysis](https://www.seangoedecke.com/text-tokens-as-image-tokens/)
  and [pxpipe](https://wavect.io/blog/text-as-image-token-savings/): rendering
  text as images can cut billed tokens for some multimodal models, but it is
  lossy and should not be used for exact code edits without a markdown fallback.

## Current Loss Profile

On this repository, using:

```bash
ai-token-saver measure . \
  --budget 3000 \
  --query "token compression evidence budget code context" \
  --mode aggressive
```

the current result is:

```text
Raw: 23,502 tokens
Pack: 2,804 tokens
Saved: 20,698 tokens (88.1%)
Token removal: 88.1%
Token retention: 11.9%
Query-term recall: 100.0%
Code-symbol recall: 100.0%
File coverage: 15/17 (88.2%)
Chunk coverage: 43/190 (22.6%)
Critical retention proxy: 95.0%
Estimated loss proxy: 5.0%
```

These are proxy metrics, not a guarantee of final answer quality. They are meant
to catch obvious compression failures: missing query terms, missing code symbols,
or overly narrow file coverage.

## Screenshot Pack Caveat

`shotpack` is for bulk reading context. Keep the generated markdown pack when
you need exact identifiers, copy/paste code, numeric values, secrets, hashes,
diffs, or legal/financial/medical text. Image OCR and vision interpretation are
model-dependent and can silently misread characters.

## Example Output

```text
Wrote context-pack.md
Raw: 23,502 tokens | Pack: 2,804 tokens | Saved: 20,698 (88.1%)
Files: 17 | Counter: heuristic
```

## Commands

### `audit`

Estimate raw and compacted token usage for files or directories.

```bash
ai-token-saver audit src README.md --budget 12000 --json
```

### `trim`

Compact one text file or stdin.

```bash
cat server.log | ai-token-saver trim - > smaller-log.txt
```

Extractive mode keeps the most relevant chunks under a budget:

```bash
ai-token-saver trim meeting-notes.md \
  --strategy extractive \
  --budget 1200 \
  --query "decisions risks next steps"
```

### `pack`

Generate a markdown context pack for AI chat tools.

```bash
ai-token-saver pack src tests README.md \
  --budget 6000 \
  --query "find bugs in token budget selection" \
  --mode aggressive \
  -o ask-ai.md
```

### `measure`

Report token savings and loss proxy metrics.

```bash
ai-token-saver measure . \
  --budget 3000 \
  --query "token compression evidence budget code context" \
  --mode aggressive
```

### `shotpack`

Generate a markdown context pack plus PNG pages.

```bash
ai-token-saver shotpack . \
  --budget 3000 \
  --query "token compression evidence budget code context" \
  --mode aggressive \
  -o shotpack
```

## What Gets Ignored

The scanner skips common heavy folders such as `.git`, `node_modules`, `dist`,
`build`, `.venv`, caches, and binary files. Individual files larger than 200 KB
are skipped by default; change that with `--max-file-bytes`.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

## License

MIT
