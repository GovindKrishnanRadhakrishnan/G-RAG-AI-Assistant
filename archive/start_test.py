import subprocess
import sys

with open('debug_out.txt', 'w', encoding='utf-8') as f:
    result = subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "9005"], stdout=f, stderr=subprocess.STDOUT)
    print("EXIT CODE:", result.returncode)
