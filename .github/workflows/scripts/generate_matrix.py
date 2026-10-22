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
        slcp_path = get_prop(desc, "projectFilePaths")
        app = slcp_path.split(".")[0]
        app_dir = str(Path(slcp_path).parent)
        slcp = Path(slcp_path).name
        use_pkg_slt = (ROOT / app_dir / "pkg.slt").is_file()
        boards = split_ws(get_prop(desc, "boardCompatibility"))

        for board in boards:
            rows.append({
                "app": app,
                "board": board,
                "app_dir": app_dir,
                "slcp": slcp,
                "use_pkg_slt": use_pkg_slt,
            })

    if not rows:
        # Avoid empty matrix which makes Actions error out
        rows = [{"app":"noop","board":"noop"}]

    matrix = {"include": rows}
    print(json.dumps(matrix, indent=2))

if __name__ == "__main__":
    main()
