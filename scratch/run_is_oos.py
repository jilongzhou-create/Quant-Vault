import subprocess
import sys

result = subprocess.run(
    [sys.executable, 'scratch/backtest_portfolio_44_is_oos.py'],
    capture_output=True, text=True, cwd=r'c:\Users\Jilong\Documents\trae_projects\trading_agent'
)
with open(r'c:\Users\Jilong\Documents\trae_projects\trading_agent\scratch\is_oos_output.txt', 'w', encoding='utf-8') as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
print("Done. Exit code:", result.returncode)
