import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('prediksi_putus_sekolah_crisp_dm_c45.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Show source code for all code cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        print(f"\n{'='*60}")
        print(f"CELL {i} [code]:")
        print(f"{'='*60}")
        print(src[:2000])
        if len(src) > 2000:
            print(f"\n... (truncated, total {len(src)} chars)")
