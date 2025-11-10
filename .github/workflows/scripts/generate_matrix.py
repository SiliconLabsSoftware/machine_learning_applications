#!/usr/bin/env python3
import os, json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
TEMPLATES_XML = ROOT / "templates.xml"

def get_prop(desc, key):
    # <properties key="boardCompatibility" value="...">
    p = desc.find(f'properties[@key="{key}"]')
    return (p.get("value") if p is not None else "").strip()

def split_ws(s):
    # boardCompatibility is space-separated: "brd2601a brd2601b"
    return [x for x in s.replace(",", " ").split() if x]

def main():
    tree = ET.parse(TEMPLATES_XML)
    root = tree.getroot()

    rows = []
    for desc in root.findall("descriptors"):
        app = get_prop(desc, "projectFilePaths").split(".")[0]     # e.g. application/voice/.../series_2.slcp -> application/voice/.../series_2    
        boards = split_ws(get_prop(desc, "boardCompatibility"))    # e.g. ["brd2601a", "brd2601b"]

        for board in boards:
            rows.append({
                "app": app,
                "board": board
            })

    if not rows:
        # Avoid empty matrix which makes Actions error out
        rows = [{"app":"noop","board":"noop"}]

    matrix = {"include": rows}
    print(json.dumps(matrix, indent=2))

if __name__ == "__main__":
    main()
