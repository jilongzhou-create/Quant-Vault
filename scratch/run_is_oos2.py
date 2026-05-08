import subprocess
import sys
import os

os.chdir(r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
result = subprocess.run(
    [sys.executable, 'scratch/backtest_portfolio_44_is_oos.py'],
    capture_output=True, text=True, timeout=600
)
outpath = r'c:\Users\Jilong\Documents\trae_projects\trading_agent\scratch\is_oos_output.txt'
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout[-15000:] if len(result.stdout) > 15000 else result.stdout)
    f.write("\n=== STDERR (last 3000) ===\n")
    f.write(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
print(f"Done. Exit code: {result.returncode}, stdout len: {len(result.stdout)}, stderr len: {len(result.stderr)}")
