# one command for the whole demo (handy in Colab / VS Code): starts both A2A servers
# in the background, waits until their agent cards are online, runs the client + orchestrator,
# then stops the servers.
#
# run:  python 07_a2a_multi_agent/run_demo.py

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVERS = {"researcher_server.py": 8001, "writer_server.py": 8002}


def wait_for(port: int, timeout: int = 60) -> None:
    url = f"http://localhost:{port}/.well-known/agent-card.json"
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=1)
            print(f"agent card online: {url}", flush=True)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"server on port {port} did not start — run it by hand to see the error")


procs = [subprocess.Popen([sys.executable, str(HERE / f)], cwd=HERE) for f in SERVERS]
try:
    for port in SERVERS.values():
        wait_for(port)
    print("\n================ plain A2A client ================", flush=True)
    subprocess.run([sys.executable, str(HERE / "a2a_client.py")], cwd=HERE, check=True)
    print("\n================ ADK orchestrator ================", flush=True)
    subprocess.run([sys.executable, str(HERE / "orchestrator.py")], cwd=HERE, check=True)
finally:
    for p in procs:
        p.terminate()
