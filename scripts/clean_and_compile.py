import re
with open('manuscript/main.tex') as f:
    tex = f.read()

tex = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', tex)
def repl_bt(m):
    s = m.group(1)
    s = s.replace(r'\%', '%').replace('%', r'\%')
    s = s.replace(r'\_', '_').replace('_', r'\_')
    return r'\texttt{' + s + '}'
tex = re.sub(r'`([^`]+)`', repl_bt, tex)

with open('manuscript/main.tex', 'w') as f:
    f.write(tex)
