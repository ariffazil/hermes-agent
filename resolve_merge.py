#!/usr/bin/env python3
"""Resolve the 8 upstream-merge conflicts on merge-test-v0.21 branch.

Doctrine: upstream refactored (decomposed) several monoliths; our backports were
already ABSORBED upstream or must be re-injected into the new structure.

Per-conflict resolution:
1.  .gitignore                 -> union (ours adds runtime-state entries)
2.  agent/prompt_builder.py    -> ours (NO_VISION_DISCLAIMER + computer_use_guidance
                                   are AAA features absent upstream)
3.  agent/system_prompt.py c1  -> ours + upstream's new imports merged (union of names)
4.  agent/system_prompt.py c2  -> ours (HEAD already contains the upstream-equivalent
                                   refactored build path; upstream side is a stale
                                   fragment of its own refactor)
5.  gateway/config.py          -> upstream stub + re-inject ASI_ARIFOS_BOT_TOKEN into
                                   the new _getenv choke-point in gateway/config.py
6.  gateway/run.py             -> ours (our _decide_image_input_mode survived; upstream
                                   moved the caller into run_inbound.py which is NOT
                                   conflicted, so our method stays)
7.  plugins/.../adapter.py     -> upstream (one-line call to _build_ptb_requests which
                                   now contains ALL our PTB tuning + improvements)
8.  tools/mcp_tool.py          -> upstream (mcp SDK imports + probe moved to
                                   mcp_tool_transport.py with our TLS/cert support)
9.  tools/send_message_tool.py -> upstream stub + re-inject anti-echo guard into
                                   tools/send_message_senders.py (_send_telegram there)
"""
import re, sys

BASE = '/tmp/hermes-merge-test'
sys.path.insert(0, BASE)

def read(p):
    with open(p) as f: return f.read()

def write(p, s):
    with open(p, 'w') as f: f.write(s)

def resolve(p, pick):
    """pick: 'ours'|'theirs' — keep that side, drop markers."""
    c = read(p)
    pat = re.compile(r'<<<<<<< HEAD\n(.*?)\n?^=======\n(.*?)^>>>>>>> origin/main\n', re.DOTALL | re.M)
    n = [0]
    def rep(m):
        n[0] += 1
        return m.group(1) if pick == 'ours' else m.group(2)
    out = pat.sub(rep, c)
    if n[0] == 0:
        print(f"  !! no conflicts matched in {p}")
        return 0
    write(p, out)
    return n[0]

def count_markers(p):
    c = read(p)
    return c.count('<<<<<<<'), c.count('>>>>>>>')

log = []

# 1. .gitignore — union: ours + theirs
p = f'{BASE}/.gitignore'
c = read(p)
m = re.search(r'<<<<<<< HEAD\n(.*?)\n^=======\n(.*?)^>>>>>>> origin/main\n', c, re.DOTALL | re.M)
assert m, "gitignore conflict vanished"
ours, theirs = m.group(1), m.group(2)
# strip a duplicated .skills_prompt_snapshot.json from theirs if ours has it
theirs_clean = '\n'.join(l for l in theirs.splitlines()
                         if l.strip() and l.strip() != '.skills_prompt_snapshot.json')
c = c[:m.start()] + ours + '\n' + theirs_clean + '\n' + c[m.end():]
write(p, c)
log.append(f".gitignore: union merged")

# 2. prompt_builder — ours (NO_VISION_DISCLAIMER + guidance fn)
n = resolve(f'{BASE}/agent/prompt_builder.py', 'ours')
log.append(f"agent/prompt_builder.py: kept ours ({n} conflict)")

# 3+4. system_prompt.py — conflict 1: union of import names; conflict 2: ours
p = f'{BASE}/agent/system_prompt.py'
c = read(p)
ms = list(re.finditer(r'<<<<<<< HEAD\n(.*?)\n^=======\n(.*?)^>>>>>>> origin/main\n', c, re.DOTALL | re.M))
assert len(ms) == 2, f"expected 2 conflicts in system_prompt, got {len(ms)}"
# conflict 1 (imports): parse names from both sides, union, re-emit in our style
head1, inc1 = ms[0].group(1), ms[0].group(2)
def imp_names(block):
    return [x.strip().rstrip(',') for x in block.splitlines() if x.strip()]
names = []
for x in imp_names(head1) + imp_names(inc1):
    if x not in names: names.append(x)
merged_imports = ',\n'.join(f'    {n}' if i % 3 else f'    {n}' for i, n in enumerate(names))
# simpler: one name per line
merged_imports = ',\n'.join(f'    {n}' for n in names)
c = c[:ms[0].start()] + merged_imports + ',\n' + c[ms[0].end():]
# conflict 2: ours
m2 = re.search(r'<<<<<<< HEAD\n(.*?)\n^=======\n(.*?)^>>>>>>> origin/main\n', c, re.DOTALL | re.M)
assert m2, "second conflict not found after first fix"
c = c[:m2.start()] + m2.group(1) + '\n' + c[m2.end():]
write(p, c)
log.append("agent/system_prompt.py: imports=union, body=ours")

# 5. gateway/config.py — upstream stub; ASI token chain goes into _getenv later
n = resolve(f'{BASE}/gateway/config.py', 'theirs')
log.append(f"gateway/config.py: took upstream stub ({n} conflict) — AAA env chain to re-inject below")

# 6. gateway/run.py — ours (image-mode method + WELL voice hooks)
n = resolve(f'{BASE}/gateway/run.py', 'ours')
log.append(f"gateway/run.py: kept ours ({n} conflict)")

# 7. telegram adapter — upstream one-liner (contains all our PTB tuning now)
n = resolve(f'{BASE}/plugins/platforms/telegram/adapter.py', 'theirs')
log.append(f"plugins/platforms/telegram/adapter.py: upstream _build_ptb_requests call ({n} conflict)")

# 8. mcp_tool.py — upstream (probe/TLS moved to mcp_tool_transport.py)
n = resolve(f'{BASE}/tools/mcp_tool.py', 'theirs')
log.append(f"tools/mcp_tool.py: upstream decomposed modules ({n} conflicts)")

# 9. send_message_tool.py — upstream stub; guard re-injected into senders file
n = resolve(f'{BASE}/tools/send_message_tool.py', 'theirs')
log.append(f"tools/send_message_tool.py: upstream stub ({n} conflict)")

print('\n'.join(log))
total_left = 0
for p in ['.gitignore','agent/prompt_builder.py','agent/system_prompt.py','gateway/config.py',
          'gateway/run.py','plugins/platforms/telegram/adapter.py','tools/mcp_tool.py',
          'tools/send_message_tool.py']:
    a,b = count_markers(f'{BASE}/{p}')
    if a or b: print(f"  !! {p}: {a} start / {b} end markers REMAIN"); total_left += a
print(f"REMAINING MARKERS: {total_left}")
