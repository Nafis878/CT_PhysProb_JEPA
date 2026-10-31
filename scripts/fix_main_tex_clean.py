import re

with open('manuscript/main.tex') as f:
    tex = f.read()

# 1. Undo the multi-line backtick corruption by restoring \_ inside \ref, \label, \eqref, \cite, and math
# Actually, let's check where \_ was inserted!
# Even simpler: let's inspect any remaining issues after restoring math/label underscores.
