#!/usr/bin/env python3
from pathlib import Path
import json, sys
root=Path('.')
inv=root/'_inventory'/'inventory.json'
if not inv.exists():
    print('inventory.json missing'); sys.exit(1)
try:
    obj=json.loads(inv.read_text(encoding='utf-8'))
    items=obj.get('items',[])
    assert isinstance(items,list) and items, 'items empty'
    for it in items:
        assert 'path' in it and 'sha256' in it, 'missing fields'
    print('verify_manifest: OK (items=',len(items),')')
except Exception as e:
    print('verify_manifest: FAIL', e); sys.exit(1)
