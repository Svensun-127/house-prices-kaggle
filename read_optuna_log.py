from pathlib import Path
path = Path('optuna_run.log')
text = path.read_text(encoding='utf-16', errors='ignore')
lines = text.splitlines()
print('TOTAL LINES', len(lines))
for i, line in enumerate(lines[:60], 1):
    print(f'{i:03d}: {line}')
print('--- tail ---')
for i, line in enumerate(lines[-20:], len(lines)-19):
    print(f'{i:03d}: {line}')
