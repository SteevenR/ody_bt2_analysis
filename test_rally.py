"""
Test script to analyze a complete rally (multi-page) with full pipeline.
This replicates the exact behavior of process_single_rally including:
- Multi-page segmentation (.0, .1, .2)
- Calibration and line detection
- Leader flag detection with tie-breaking
- Name resolution with translations

Usage: python test_rally.py <event_date> <rally_id>
Example: python test_rally.py 2025-12-17 24
Example (totals): python test_rally.py 2025-12-17 total
"""

import sys
import json
from pathlib import Path
from lib.config import DATA_INPUT_DIR, DATA_OUTPUT_DIR
from lib.ocr_utils import init_reader, init_gpu, GPU_STATUS
from lib.name_resolution import load_translations_store, resolve_with_translation, save_translations_store
from lib.rally_extraction import extract_rally_participants
from lib.leader_detection import detect_leader_with_tiebreaker, detect_leader_fallback


def find_rally_pages(event_dir: Path, rally_id: str) -> list[Path]:
    """Find all pages for a rally (.0, .1, .2)."""
    if rally_id == "total":
        pattern = "total."
    else:
        pattern = f"{rally_id}."
    
    pages = []
    for f in sorted(event_dir.iterdir()):
        if f.stem.startswith(pattern) and f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            pages.append(f)
    
    return pages


def analyze_rally(event_date: str, rally_id: str, debug_mode=True):
    """Analyze a complete rally with full pipeline."""
    
    print("\n" + "="*80)
    print(f"RALLY ANALYSIS TEST")
    print("="*80)
    print(f"Event: {event_date}")
    print(f"Rally: {rally_id}")
    print("="*80 + "\n")
    
    # Initialize GPU and OCR
    print("[INFO] Initializing GPU...")
    init_gpu()
    print(f"[INFO] GPU Status: {GPU_STATUS}")
    
    print("[INFO] Initializing OCR reader...")
    reader = init_reader()
    
    # Find rally pages
    event_dir = DATA_INPUT_DIR / event_date
    if not event_dir.exists():
        print(f"[ERROR] Event directory not found: {event_dir}")
        return
    
    pages = find_rally_pages(event_dir, rally_id)
    if not pages:
        print(f"[ERROR] No pages found for rally {rally_id}")
        return
    
    print(f"[INFO] Found {len(pages)} pages:")
    for p in pages:
        print(f"  - {p} (full path)")
    print()
    
    # Load translations store
    trans_store = load_translations_store(DATA_OUTPUT_DIR / "player_translations.json")
    print(f"[INFO] Loaded {len(trans_store.get('players', {}))} known players\n")
    
    # Extract participants
    rally_id_int = -1 if rally_id == "total" else int(rally_id)
    event_id = event_date
    
    print("[INFO] Extracting participants...\n")
    participants_raw = extract_rally_participants(
        reader, pages,
        ocr_logger=None,
        event_id=event_id,
        rally_id=rally_id_int,
        debug_mode=debug_mode
    )
    
    if not participants_raw:
        print("[ERROR] No participants extracted")
        return
    
    print(f"\n[INFO] Extracted {len(participants_raw)} participants (raw)\n")
    
    # Debug: print participants before dedup
    if debug_mode and len(participants_raw) > 11:
        print("[DEBUG] Raw participants:")
        for p in participants_raw:
            print(f"  - rank {p['rank']:2d} {p.get('name', '???'):20s} damage={p.get('damage', 0):10,} from {p.get('source_file', '').split(chr(92))[-1]}")
    
    # Deduplicate participants
    seen = set()
    participants_unique = []
    for p in participants_raw:
        key = (p["name"], p["rank"])
        if key not in seen:
            seen.add(key)
            participants_unique.append(p)
    
    print(f"[INFO] {len(participants_unique)} unique participants\n")
    
    # Detect leader
    print("[INFO] Detecting leader...\n")
    flagged = [p for p in participants_unique if p.get("has_flag")]
    
    print(f"[INFO] Found {len(flagged)} participants with flags:")
    for p in flagged:
        print(f"  - {p['name']} (rank {p['rank']}, damage={p['damage']:,}, flag_score={p.get('final_score', 0):.3f}, page={p.get('page_number', 0)})")
    print()
    
    leader_name_obs = None
    if flagged:
        if len(flagged) == 1:
            leader_name_obs = flagged[0]["name"]
            print(f"[INFO] Leader detected (single flag): {leader_name_obs}")
        else:
            # Tie-breaking: prefer higher damage, then earlier page, then lower rank
            leader_name_obs = detect_leader_with_tiebreaker(flagged)
            print(f"[INFO] Leader detected (tie-breaking): {leader_name_obs}")
            print(f"[INFO] Tie-breaker logic: max(damage, -page_number, -rank)")
    else:
        print(f"[INFO] Leader: Not found (no flagged participants)")
    
    print()
    
    # Resolve names
    print("[INFO] Resolving player names...\n")
    resolved_players = []
    trans_added = False
    
    for p in sorted(participants_unique, key=lambda x: x.get("damage", 0), reverse=True):
        pid, cname, lang, matched_by, added = resolve_with_translation(trans_store, p["name"])
        if added:
            trans_added = True
        
        is_leader = (p["name"] == leader_name_obs)
        
        resolved_players.append({
            "canonical_id": pid,
            "name": cname,
            "name_original": p["name"],
            "matched_by": matched_by,
            "language_detected": lang,
            "damage": p["damage"],
            "rank": p["rank"],
            "is_leader": is_leader,
            "has_flag": p.get("has_flag", False),
            "flag_score": p.get("final_score", 0)
        })
        
        flag_marker = "[F]" if p.get("has_flag") else "   "
        leader_marker = "[L]" if is_leader else "   "
        dmg = p.get("damage", 0) or 0
        cname_display = cname or "???"
        matched_display = matched_by or "?"
        print(f"{flag_marker}{leader_marker} [{p['rank']:2d}] {cname_display:30s} {dmg:12,} ({matched_display:6s}) {pid}")
    
    # Save translations if modified
    if trans_added:
        save_translations_store(DATA_OUTPUT_DIR / "player_translations.json", trans_store)
        print(f"\n[INFO] Updated player translations")
    
    # Count participants per file
    from pathlib import Path
    file_counts = {}
    for p in participants_unique:
        src = p.get("source_file", "unknown")
        # Normalize path for comparison
        if src != "unknown":
            src_norm = str(Path(src).resolve())
            file_counts[src_norm] = file_counts.get(src_norm, 0) + 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Event: {event_date}")
    print(f"Rally: {rally_id}")
    print(f"Pages: {len(pages)}")
    print("Files used:")
    for p in pages:
        p_norm = str(p.resolve())
        count = file_counts.get(p_norm, 0)
        print(f"  - {p} ({count} players)")
    print(f"Participants: {len(resolved_players)}")
    leader_display = leader_name_obs if leader_name_obs else "Not found"
    print(f"Leader: {leader_display}")
    print(f"Flagged players: {len(flagged)}")
    print("="*80 + "\n")
    
    # Return result for programmatic use
    return {
        "event": event_date,
        "rally_id": rally_id,
        "leader": leader_name_obs,
        "participants": resolved_players,
        "pages": [str(p) for p in pages]
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_rally.py <event_date> <rally_id>")
        print("Example: python test_rally.py 2025-12-17 24")
        print("Example (totals): python test_rally.py 2025-12-17 total")
        sys.exit(1)
    
    event_date = sys.argv[1]
    rally_id = sys.argv[2]
    
    analyze_rally(event_date, rally_id, debug_mode=True)


if __name__ == "__main__":
    main()
