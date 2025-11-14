#!/usr/bin/env python3
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
man=root/"MANIFEST-SSOT.md"
if not man.exists():
    print('MANIFEST-SSOT.md missing'); sys.exit(0)
rows=[]
for ln in man.read_text(encoding='utf-8',errors='ignore').splitlines():
    s=ln.strip()
    if s.startswith('|') and not s.startswith('|---'):
        cols=[c.strip() for c in s.strip('|').split('|')]
        if len(cols)>=4 and cols[0] != 'path':
            rows.append((cols[0],[a.strip() for a in (cols[3] or '').split(',') if a.strip()]))
missing=[]
for path,anchors in rows:
    p=(root/path.lstrip('/'))
    if not p.exists():
        missing.append((path,'<file-missing>')); continue
    data=p.read_text(encoding='utf-8',errors='ignore')
    for a in anchors:
        if a and a not in data: missing.append((path,a))
if missing:
    for path,a in missing: print(f'MISSING {path} :: {a}')
    sys.exit(1)
print('OK')
