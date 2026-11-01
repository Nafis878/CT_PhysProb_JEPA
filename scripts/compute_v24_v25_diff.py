#!/usr/bin/env python3
"""
Compute unified diff between PREREGISTRATION_V2.md (v2.4) and PREREGISTRATION_V2.md (v2.5).
"""

import difflib
import subprocess

# Get v2.4 text from git show preregistration-v2.4:PREREGISTRATION_V2.md
cmd = ['C:/Program Files/Git/cmd/git.exe', 'show', 'preregistration-v2.4:PREREGISTRATION_V2.md']
v24_text = subprocess.check_output(cmd, encoding='utf-8')

with open('PREREGISTRATION_V2.md', 'r', encoding='utf-8') as f:
    v25_text = f.read()

diff = list(difflib.unified_diff(
    v24_text.splitlines(keepends=True),
    v25_text.splitlines(keepends=True),
    fromfile='PREREGISTRATION_V2.md (v2.4)',
    tofile='PREREGISTRATION_V2.md (v2.5)',
    n=2
))

print(''.join(diff))
