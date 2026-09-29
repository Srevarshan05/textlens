# Benchmarking, profiling and visual debugging

Model cards report results on public benchmarks. Your invoices, scans and
camera frames are different. TextLens measures models **on your data, on your
hardware** — and labels every number as a TextLens measurement.

## Benchmark

```bash
textlens benchmark ./dataset                              # every installed local model
textlens benchmark ./dataset --models ppocrv6-small,glm-ocr
textlens benchmark ./dataset --profiles fast,balanced     # compare routing profiles
textlens benchmark ./dataset -o report.json
```

### Dataset layout

```text
dataset/
  images/  invoice-001.png  scan-002.pdf …
  ground_truth/  invoice-001.txt  scan-002.txt …
```

or siblings (`a.png` + `a.txt`), or `manifest.jsonl`:

```json
{"file": "images/a.png", "text": "expected text", "table": [["Item", "Qty"], ["Apple", "3"]]}
```

### Metrics

| Metric | Meaning |
|---|---|
| CER / WER | character / word edit distance ÷ reference length (micro-averaged over the dataset) |
| Exact | share of samples matching exactly after whitespace normalisation |
| p50 / p95 | per-sample latency (after a warm-up sample) |
| pages/s | throughput over the run |
| Peak RAM / VRAM | process peak during the run |
| table cell accuracy | share of reference cells reproduced at the same position (+ shape match) |
| confidence gap | mean confidence of good reads (CER ≤ 5%) minus bad ones — is confidence informative? |

Every report embeds an environment fingerprint (TextLens, Python, CPU, GPU,
runtimes, timestamp) and the note *"TextLens measurements on this dataset and
machine; not official model benchmarks"*.

### Measurements feed routing

Latencies are saved to `$TEXTLENS_HOME/measurements.json`. The router then
uses measured per-page latency instead of its built-in speed tiers — for
`latency_budget_ms` and for scoring. `--no-save` disables this.

## Profile one document

```bash
textlens profile report.pdf --repeat 3
```

Reports the cold run (with model loading), warm runs, per-page time, model
load time, time per stage (`inspect`, `extract`, `render`, `detect`,
`recognize`, `generate`), peak RAM and this process's peak GPU memory.

## See what TextLens saw

```bash
textlens inspect report.pdf                       # page kinds, OCR reasons, table/math signals
textlens inspect report.pdf --overlay overlays/   # annotated page images
```

Overlays colour blocks by source (native green, OCR blue, VLM purple, fused
orange, tables red), number them in reading order, mark low-confidence blocks,
and print the routing decision in a header strip.

## Tips for fair comparisons

- Use documents like your production traffic, including hard ones.
- Compare profiles too: `balanced` may beat any single model by routing.
- Run on the target hardware (a laptop number says little about a Pi).
- Re-run after model or TextLens upgrades; keep reports next to the config hash.
