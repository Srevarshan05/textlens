"""
textlens.sdk
────────────
High-level developer SDK for TextLens OCR.
Supports image, PDF, table, formula, and structured JSON extraction
using zai-org/GLM-OCR with auto GPU acceleration & dynamic runtime device switching.
"""

from __future__ import annotations

import os
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

from textlens.hardware import get_hardware_info, HardwareInfo


class TextLens:
    """
    Main developer client for TextLens OCR.

    Examples
    --------
    >>> from textlens import TextLens
    >>> ocr = TextLens()  # Auto-detects GPU (CUDA) or CPU
    >>> text = ocr.read("sample.png")
    >>> pages = ocr.read_pdf("document.pdf")
    >>> table = ocr.extract_table("invoice.jpg")
    >>> ocr.switch_device("cpu")  # Dynamic device switching
    """

    def __init__(
        self,
        model_id: str = "zai-org/GLM-OCR",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        auto_load: bool = True
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
        """
        self.model_id = model_id
        
        # Hardware auto-detection
        hw = get_hardware_info()
        if device is None:
            self.device = "cuda" if hw.gpu_available else "cpu"
        else:
            self.device = device.lower().strip()

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if (self.device == "cuda" and hw.gpu_available) else torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.processor = None
        self.model = None
        self._is_loaded = False

        if auto_load:
            self.load()

    def load(self) -> None:
        """Load HuggingFace processor and model into memory."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers module is required. Install via `pip install transformers`"
            )

        print(f"[TextLens] Loading model '{self.model_id}' onto device: {self.device.upper()} ...")

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.device == "cuda" and torch.cuda.is_available():
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

        self.model.eval()
        self._is_loaded = True
        
        active_dev = next(self.model.parameters()).device
        print(f"[TextLens] Model successfully loaded on: {active_dev}")

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded into memory."""
        return self._is_loaded

    @property
    def hardware(self) -> HardwareInfo:
        """Retrieve current system hardware details."""
        return get_hardware_info()

    def switch_device(self, target_device: str) -> str:
        """
        Dynamically switch runtime model execution between 'cuda' (GPU) and 'cpu'.

        Parameters
        ----------
        target_device : 'cuda' or 'cpu'

        Returns
        -------
        str
            Status message describing the active execution device.
        """
        if not self._is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        target = target_device.lower().strip()
        if target in ("gpu", "cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available on this system.")
            self.model.to("cuda")
            self.device = "cuda"
            msg = f"Device switched to CUDA GPU ({torch.cuda.get_device_name(0)})"
        else:
            self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            msg = "Device switched to CPU"

        print(f"[TextLens] {msg}")
        return msg

    def read(
        self,
        image_source: Union[str, Path, Image.Image],
        prompt: str = "Text Recognition:",
        max_new_tokens: int = 512
    ) -> str:
        """
        Read text from an image (local file path, remote HTTP/HTTPS URL, or PIL Image object).

        Parameters
        ----------
        image_source : str, Path, or PIL Image
            Path to image, http/https URL, or PIL Image object.
        prompt : str
            Instruction prompt to guide model output style.
        max_new_tokens : int
            Maximum generation token length.

        Returns
        -------
        str
            Extracted text content.
        """
        if not self._is_loaded:
            self.load()

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

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()

    def read_pdf(
        self,
        pdf_source: Union[str, Path],
        prompt: str = "Text Recognition:",
        scale: float = 2.0,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a multi-page PDF document page by page.

        Parameters
        ----------
        pdf_source : str or Path
            Path to PDF file.
        prompt : str
            Instruction prompt.
        scale : float
            Render resolution multiplier (higher scale = sharper OCR).
        max_pages : int, optional
            Limit number of pages to process.

        Returns
        -------
        List[Dict[str, Any]]
            List of page objects: [{'page': 1, 'text': '...'}, ...]
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx in range(pages_to_process):
                page = pdf_doc[idx]
                pil_img = page.render(scale=scale).to_pil().convert("RGB")
                tmp_path = Path(tmp_dir) / f"page_{idx + 1}.png"
                pil_img.save(tmp_path)

                page_text = self.read(tmp_path, prompt=prompt)
                results.append({
                    "page": idx + 1,
                    "total_pages": total_pages,
                    "text": page_text
                })

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
        """Process a list of images or URLs sequentially."""
        return [self.read(src, prompt=prompt) for src in sources]
