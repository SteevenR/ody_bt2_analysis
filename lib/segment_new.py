"""New segmentation logic: segment all lines until end of image using calibrated line height"""
import cv2
from pathlib import Path
from .config import DATA_OUTPUT_DIR
from .image_processing import find_first_rank_ocr


def segment_lines_dynamic(img_up, line_height_px, rank_y_positions: dict, img_width, page_stem: str, reader=None, is_totals=False, debug_mode=False, event_id="", rally_id=0):
    """Segment image into lines based on calibrated line height.
    
    Once calibrated, segment ALL lines from image top to bottom using the line_height_px,
    not assuming a fixed number of ranks.
    
    Args:
        img_up: Upsampled image
        line_height_px: Calibrated line height
        rank_y_positions: Dict of rank -> y_position from calibration
        img_width: Image width
        page_stem: Page identifier (.0, .1, .2)
        reader: OCR reader (for OCR-based first rank detection)
        is_totals: Whether this is totals (vs rally)
        debug_mode: Enable debug output
        event_id: Event ID for filenames
        rally_id: Rally ID for filenames
    
    Returns:
        List of (y1, y2, row_img, presumed_rank) tuples
    """
    h, w = img_up.shape[:2]
    half_height = int(line_height_px / 2)
    lines = []
    
    if page_stem.endswith(".0"):
        # .0 page: start from rank 1, segment until bottom
        rank_1_y = rank_y_positions.get(1, 0)
        current_y = rank_1_y - half_height if rank_1_y > 0 else 0
        rank = 1
        
        while current_y < h and rank <= 15:  # Safety limit
            y1 = max(0, int(current_y))
            y2 = min(h, int(current_y + line_height_px))
            row_img = img_up[y1:y2, 0:w]
            
            if row_img.size > 0:
                lines.append((y1, y2, row_img, rank))
            
            current_y += line_height_px
            rank += 1
        
        if debug_mode:
            print(f"[DEBUG] .0: segmented {len(lines)} lines (img_height={h}, line_height={line_height_px:.1f})")
            for y1, y2, _, presumed_rank in lines:
                print(f"  - presumed_rank {presumed_rank}: y1={y1}, y2={y2}, height={y2-y1}")
    
    elif page_stem.endswith(".2"):
        # .2 page: small page, segment from top
        if is_totals:
            # Totals .2: use pre-calculated ranks
            for rank in sorted(rank_y_positions.keys()):
                y_center = rank_y_positions[rank]
                y1 = max(0, int(y_center - half_height))
                y2 = min(h, int(y_center + half_height))
                row_img = img_up[y1:y2, 0:w]
                if row_img.size > 0:
                    lines.append((y1, y2, row_img, rank))
        else:
            # Rally .2: segment from top, starting at rank 12
            current_y = 0
            rank = 12
            
            while current_y < h and rank <= 15:
                y1 = max(0, int(current_y))
                y2 = min(h, int(current_y + line_height_px))
                row_img = img_up[y1:y2, 0:w]
                
                if row_img.size > 0:
                    lines.append((y1, y2, row_img, rank))
                
                current_y += line_height_px
                rank += 1
        
        if debug_mode:
            print(f"[DEBUG] .2: segmented {len(lines)} lines (img_height={h}, line_height={line_height_px:.1f})")
            for y1, y2, _, presumed_rank in lines:
                print(f"  - presumed_rank {presumed_rank}: y1={y1}, y2={y2}, height={y2-y1}")
    
    else:  # .1 page
        # .1 page: find first rank and segment until bottom
        first_rank_y = None
        if reader:
            first_rank_y_val, _ = find_first_rank_ocr(reader, img_up, line_height_px, 5)
            first_rank_y = first_rank_y_val
        
        current_y = first_rank_y - half_height if first_rank_y is not None else 0
        rank = 5  # .1 starts at rank 5
        
        if debug_mode:
            print(f"[DEBUG] .1: first_rank_y={first_rank_y}, starting current_y={current_y}, line_height={line_height_px:.1f}")
        
        while current_y < h and rank <= 15:
            y1 = max(0, int(current_y))
            y2 = min(h, int(current_y + line_height_px))
            row_img = img_up[y1:y2, 0:w]
            
            if row_img.size > 0:
                lines.append((y1, y2, row_img, rank))
                if debug_mode:
                    print(f"[DEBUG]   presumed_rank {rank}: y1={y1}, y2={y2}, height={y2-y1}")
            
            current_y += line_height_px
            rank += 1
        
        if debug_mode:
            print(f"[DEBUG] .1: segmented {len(lines)} lines")
    
    return lines
