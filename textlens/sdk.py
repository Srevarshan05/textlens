"""
textlens.sdk
────────────
High-level developer SDK for TextLens OCR.
Supports image, PDF, table, formula, and structured JSON extraction
using zai-org/GLM-OCR with real-time progress, CUDA GPU acceleration, and optimized inference.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import torch
from PIL import Image

try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False

try:
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from textlens.hardware import get_hardware_info, is_cuda_available, HardwareInfo
from textlens.dependencies import check_dependencies, ensure_dependencies
from textlens.progress import ProgressTracker, print_step

logger = logging.getLogger("textlens.sdk")


class TextLens:
    """
    Main developer client for TextLens OCR framework.

    Examples
    --------
    >>> from textlens import TextLens
    >>> ocr = TextLens()  # Auto-detects GPU (CUDA) or CPU
    >>> text = ocr.read("sample.png")
    >>> pages = ocr.read_pdf("document.pdf")
    >>> table = ocr.extract_table("invoice.jpg")
    >>> ocr.serve(port=8000)  # Launch REST API server
    """

    def __init__(
        self,
        model_id: str = "zai-org/GLM-OCR",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        auto_load: bool = True,
        auto_fix_dependencies: bool = True,
        show_progress: bool = True
    ) -> None:
        """
        Initialize TextLens engine instance.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID (defaults to 'zai-org/GLM-OCR').
        device : str, optional
            'cuda' or 'cpu'. If None, auto-selects 'cuda' when available, else 'cpu'.
        torch_dtype : torch.dtype, optional
            Precision dtype. Defaults to float16 for CUDA, float32 for CPU.
        auto_load : bool
            Whether to load model weights into memory immediately on initialization.
        auto_fix_dependencies : bool
            Automatically install missing dependencies if needed.
        show_progress : bool
            Whether to display real-time terminal progress indicators.
        """
        self.model_id = model_id
        self.show_progress = show_progress
        
        # Check environment dependencies
        ensure_dependencies(auto_install=auto_fix_dependencies, verbose=show_progress)

        # Hardware auto-detection
        hw = get_hardware_info()
        if device is None:
            self.device = "cuda" if hw.gpu_available else "cpu"
        else:
            self.device = device.lower().strip()

        if self.device == "cuda" and is_cuda_available():
            torch.backends.cudnn.benchmark = True

        if show_progress and self.device == "cpu":
            print(
                "[TextLens Notice] Running engine on CPU. "
                "Note: CPU execution is functional, but processing large documents "
                "or high-resolution PDFs will be slower compared to CUDA GPU acceleration."
            )

        if torch_dtype is None:
            if self.device == "cuda" and hw.gpu_available:
                # Use bfloat16 if supported, else float16
                self.torch_dtype = torch.bfloat16 if (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.processor = None
        self.model = None
        self._is_loaded = False

        if auto_load:
            self.load()

    def load(self) -> None:
        """Load HuggingFace processor and model into VRAM/RAM with real-time progress."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers module is required to run GLM-OCR model. "
                "Install via `pip install transformers accelerate`"
            )

        progress = ProgressTracker(100, desc=f"Loading Model ({self.device.upper()})")
        progress.update(20, "Initializing Processor...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        progress.update(50, "Loading Model Weights into Memory...")

        if self.device == "cuda" and is_cuda_available():
            self.model = GlmOcrForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=self.torch_dtype,
            )
        else:
            self.model = GlmOcrForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
            )

        progress.update(85, "Optimizing Model Evaluation State...")
        self.model.eval()
        self._is_loaded = True

        active_dev = next(self.model.parameters()).device
        progress.complete(f"Loaded on {active_dev}")

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded into memory."""
        return self._is_loaded

    @property
    def hardware(self) -> HardwareInfo:
        """Retrieve current system hardware details."""
        return get_hardware_info()

    def is_cuda(self) -> bool:
        """Check if active engine device is CUDA GPU."""
        return self.device == "cuda" and is_cuda_available()

    def switch_device(self, target_device: str) -> str:
        """
        Dynamically switch runtime model execution between 'cuda' (GPU) and 'cpu'.
        """
        if not self._is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        target = target_device.lower().strip()
        if target in ("gpu", "cuda"):
            if not is_cuda_available():
                raise RuntimeError("CUDA GPU is not available on this system.")
            self.model.to("cuda")
            self.device = "cuda"
            msg = f"Device switched to CUDA GPU ({torch.cuda.get_device_name(0)})"
        else:
            self.model.to("cpu")
            self.device = "cpu"
            if is_cuda_available():
                torch.cuda.empty_cache()
            msg = "Device switched to CPU"

        print(f"[TextLens] {msg}")
        return msg

    def read(
        self,
        image_source: Union[str, Path, Image.Image],
        prompt: str = "Text Recognition:",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """
        Read text from an image with real-time progress and CUDA inference acceleration.
        """
        if not self._is_loaded:
            self.load()

        start_time = time.time()
        progress = ProgressTracker(100, desc="OCR Inference") if self.show_progress else None

        if progress:
            progress.update(20, "Preparing Image Inputs...")

        # Handle input image source types
        if isinstance(image_source, Image.Image):
            content_item = {"type": "image", "image": image_source.convert("RGB")}
        elif isinstance(image_source, (str, Path)):
            src_str = str(image_source)
            if src_str.startswith(("http://", "https://")):
                content_item = {"type": "image", "url": src_str}
            else:
                p = Path(src_str)
                if not p.exists():
                    raise FileNotFoundError(f"Image file not found: {p}")
                content_item = {"type": "image", "image": Image.open(p).convert("RGB")}
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        messages = [
            {
                "role": "user",
                "content": [content_item, {"type": "text", "text": prompt}],
            }
        ]

        if progress:
            progress.update(50, "Tokenizing Multimodal Input...")

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        if progress:
            progress.update(75, f"Running Generation on {self.device.upper()}...")

        # Optimized Inference Mode
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        result_text = self.processor.decode(generated[0], skip_special_tokens=True).strip()

        elapsed = round(time.time() - start_time, 3)
        if progress:
            progress.complete(f"⚡ Completed in {elapsed}s")

        return result_text

    def read_pdf(
        self,
        pdf_source: Union[str, Path],
        prompt: str = "Text Recognition:",
        scale: float = 2.0,
        max_pages: Optional[int] = None,
        max_new_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a multi-page PDF document with real-time per-page progress.
        """
        if not PDFIUM_AVAILABLE:
            raise ImportError(
                "pypdfium2 is required for PDF OCR. Install via `pip install pypdfium2`"
            )

        pdf_path = Path(pdf_source)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_doc = pdfium.PdfDocument(pdf_path)
        total_pages = len(pdf_doc)
        pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)

        results = []
        start_time = time.time()

        print(f"\n[TextLens PDF] Processing PDF Document ({pages_to_process} pages)...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx in range(pages_to_process):
                page_num = idx + 1
                progress = ProgressTracker(100, desc=f"PDF Page {page_num}/{pages_to_process}")
                progress.update(25, "Rendering Page...")

                page = pdf_doc[idx]
                pil_img = page.render(scale=scale).to_pil().convert("RGB")
                tmp_path = Path(tmp_dir) / f"page_{page_num}.png"
                pil_img.save(tmp_path)

                progress.update(60, "Extracting Text...")
                page_text = self.read(tmp_path, prompt=prompt, max_new_tokens=max_new_tokens)
                results.append({
                    "page": page_num,
                    "total_pages": total_pages,
                    "text": page_text
                })

                progress.complete(f"Page {page_num} Complete")

        total_elapsed = round(time.time() - start_time, 3)
        print(f"⚡ [TextLens PDF] All {pages_to_process} pages processed in {total_elapsed}s!")
        return results

    def extract_table(self, image_source: Union[str, Path, Image.Image]) -> str:
        """Extract table from image formatted directly as Markdown."""
        return self.read(image_source, prompt="Extract all table contents and output as formatted Markdown:")

    def extract_formula(self, image_source: Union[str, Path, Image.Image]) -> str:
        """Extract math formulas from image rendered in LaTeX syntax."""
        return self.read(image_source, prompt="Extract mathematical formulas from this image into LaTeX syntax:")

    def extract_json(self, image_source: Union[str, Path, Image.Image], schema: Optional[str] = None) -> str:
        """Extract key document fields formatted as structured JSON."""
        prompt = "Extract all key information from this document into a clean, structured JSON object:"
        if schema:
            prompt += f"\nFollow this exact JSON schema format:\n{schema}"
        return self.read(image_source, prompt=prompt)

    def batch_read(
        self,
        sources: List[Union[str, Path, Image.Image]],
        prompt: str = "Text Recognition:"
    ) -> List[str]:
        """Process a list of images or URLs sequentially with progress tracking."""
        total = len(sources)
        results = []
        for i, src in enumerate(sources):
            print(f"\n[TextLens Batch] Item {i + 1}/{total}")
            results.append(self.read(src, prompt=prompt))
        return results

    def serve(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
        """Launch REST API server using this pre-loaded TextLens instance."""
        from textlens.server import serve
        serve(host=host, port=port, reload=reload, engine=self)
