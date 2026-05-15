from pathlib import Path
path = Path('optuna_run.log')
text = path.read_text(encoding='utf-16', errors='ignore')
lines = text.splitlines()
for i in range(max(0, len(lines)-40), len(lines)):
    print(f'{i+1:03d}: {lines[i]}')
