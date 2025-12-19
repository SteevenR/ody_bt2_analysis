"""OCR initialization and basic text extraction utilities"""
import cv2
from pathlib import Path

try:
    import torch
    import easyocr
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# GPU status globals
GPU_AVAILABLE = False
GPU_STATUS = "CPU-only"


def init_gpu():
    """Detect and initialize GPU (ROCm for AMD)."""
    global GPU_AVAILABLE, GPU_STATUS
    if not TORCH_AVAILABLE:
        GPU_STATUS = "CPU-only (torch not available)"
        return
    try:
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            GPU_STATUS = f"CUDA available, {torch.cuda.device_count()} device(s)"
            torch.cuda.init()
        elif hasattr(torch.version, "hip") and torch.version.hip is not None:
            GPU_AVAILABLE = True
            GPU_STATUS = f"ROCm/HIP available (version {torch.version.hip})"
        else:
            GPU_STATUS = "CPU-only (no GPU detected)"
    except Exception as e:
        GPU_STATUS = f"CPU-only (error during GPU init: {e})"


def init_reader_gpu():
    """Initialize EasyOCR Reader configured for GPU if available."""
    try:
        reader = easyocr.Reader(["fr", "en"], gpu=GPU_AVAILABLE, verbose=False)
        return reader
    except (ValueError, RuntimeError) as e:
        return easyocr.Reader(["fr", "en"], gpu=False, verbose=False)


def init_reader():
    """Initialize OCR reader."""
    return init_reader_gpu()


def ocr_image_lines(reader, image_path: Path):
    """Run OCR on entire image and return lines."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    try:
        return reader.readtext(img, detail=1)
    except Exception as e:
        return []
