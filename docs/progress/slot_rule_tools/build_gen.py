"""Rebuild repo/main/retrieval/generation.py = snapshot generation.py + tools/rule_block.txt."""
from pathlib import Path
here = Path(__file__).resolve().parent
snap = (here.parents[1] / 'snapshot/main/retrieval/generation.py').read_text()
block = (here / 'rule_block.txt').read_text()
snap = snap.replace("import json\nimport os\nfrom pathlib import Path\n", "import json\nimport math\nimport os\nfrom pathlib import Path\nimport re\n", 1)
old_start = "\ndef answer(context, *, generator='gemini', client=None, env_file='.env'):\n"
old_end = "    if generator == 'extractive':\n"
i = snap.index(old_start); j = snap.index(old_end, i) + len(old_end)
out = snap[:i] + block + snap[j:]
(here.parent / 'repo/main/retrieval/generation.py').write_text(out)
print('built')
