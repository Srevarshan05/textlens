# Raspberry Pi

**Use a 64-bit OS.** ONNX Runtime publishes aarch64 wheels but no 32-bit ARM
(armv7l) wheels. Raspberry Pi OS 64-bit or Ubuntu Server arm64 on a Pi 4 or
Pi 5 works.

```bash
uname -m            # must print aarch64
python3 -m venv ~/textlens && source ~/textlens/bin/activate
pip install --upgrade pip
pip install textlens-ocr
textlens setup --profile edge --yes
textlens doctor
```

`textlens doctor` reports `raspberry-pi` as the platform and recommends the
`edge` profile; `auto` selects it automatically on the device.

## Tips

- **Memory**: a 2 GB Pi runs PP-OCR comfortably. On 1 GB boards use
  `OCR(profile="edge", low_memory=True)` and keep PDF DPI at 150.
- **Threads**: the Pi 4/5 has 4 cores; the default uses all four. Leave one
  free if the device does other work: `OCR(threads=3)`.
- **Cooling**: sustained OCR throttles an uncooled Pi. Use a heatsink/fan and
  check `vcgencmd measure_temp`.
- **Storage**: keep models on the SD card or SSD (`TEXTLENS_MODELS_DIR`); an
  SSD speeds up model loading.
- **Camera pipelines**: pass frames as numpy arrays (`OCR()(frame)`), no file
  writes needed.

## As a service

```bash
pip install "textlens-ocr[server]"
TEXTLENS_API_KEYS=change-me textlens serve --host 0.0.0.0 --profile edge --warmup
```

or with Docker (see [Docker](docker.md); the CPU image builds for arm64).

## 32-bit OS

If you must stay on 32-bit Raspberry Pi OS, `pip install textlens-ocr` still
installs (ONNX Runtime is skipped on armv7), and native PDF extraction works,
but OCR needs a custom ONNX Runtime build. Migrating to a 64-bit OS is the
supported path.
