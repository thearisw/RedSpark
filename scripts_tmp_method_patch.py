"""Add a training-method choice (layout search by default, network optional). Idempotent."""
import sys, re, pathlib
root = pathlib.Path(sys.argv[1])
def patch(rel, pairs):
    p = root / rel; s = p.read_text(encoding="utf-8"); orig = s
    for old, new, marker in pairs:
        if marker in s:
            continue
        if old not in s:
            sys.exit(f"{rel}: anchor not found: {old[:60]!r}")
        s = s.replace(old, new, 1)
    if s != orig:
        p.write_text(s, encoding="utf-8"); print("patched", rel)
    else:
        print("already", rel)

patch("redspark/server.py", [
    ("from .designer import Trainer, TruthTableTask, evaluate",
     "from .designer import LayoutTrainer, Trainer, TruthTableTask, evaluate", "LayoutTrainer, Trainer"),
    ("            self.trainer = Trainer(task, max_blocks=max_blocks)",
     "            # \"layout\" (default) evolves layouts directly; \"network\" evolves a block-placing network.\n"
     "            cls = Trainer if msg.get(\"method\") == \"network\" else LayoutTrainer\n"
     "            self.trainer = cls(task, max_blocks=max_blocks)", "cls = Trainer if msg.get"),
    ('"max_blocks": 12,', '"max_blocks": 12, "method": "layout" | "network",', '"method": "layout" | "network"'),
])
patch("web/src/train.ts", [
    ('max_blocks: Number($<HTMLInputElement>("maxBlocks").value),',
     'max_blocks: Number($<HTMLInputElement>("maxBlocks").value),\n      method: $<HTMLSelectElement>("trainMethod").value,',
     'trainMethod'),
])
patch("web/index.html", [
    ('<div class="row"><label>Max blocks</label><input id="maxBlocks" type="number" min="1" max="40" value="12" /></div>',
     '<div class="row"><label>Max blocks</label><input id="maxBlocks" type="number" min="1" max="40" value="12" /></div>\n'
     '        <div class="row"><label>Method</label><select id="trainMethod" title="How new designs are made">'
     '<option value="layout" selected>Evolve layouts (fast)</option>'
     '<option value="network">Evolve a network (old)</option></select></div>',
     'id="trainMethod"'),
])
