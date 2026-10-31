import json
from pathlib import Path

p = Path(r"c:\Users\UseR\Downloads\Train_Eval_CT_PhysProb_JEPA_TriBenchmark_Q1 (1).ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        outputs = cell.get('outputs', [])
        print(f"\n--- Code Cell {idx} (Outputs: {len(outputs)}) ---")
        src_preview = "".join(cell.get('source', []))[:120].strip().replace('\n', ' ')
        print(f"Source: {src_preview}...")
        for o_idx, out in enumerate(outputs):
            out_type = out.get('output_type', '')
            if 'text' in out:
                txt = "".join(out['text'])
                print(f"  [Text Output {o_idx}]:")
                print(txt[:2500])
                if len(txt) > 2500:
                    print(f"  ... (+ {len(txt) - 2500} bytes truncated)")
            elif 'ename' in out:
                print(f"  [Error {o_idx}]: {out.get('ename')}: {out.get('evalue')}")
            elif 'data' in out:
                keys = list(out['data'].keys())
                print(f"  [Data Output {o_idx}]: keys={keys}")
                if 'text/plain' in out['data']:
                    print("   " + "".join(out['data']['text/plain'])[:500])
