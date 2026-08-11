#!/bin/bash
# Run over ssh on a fresh pod, after the repo is rsync'd.
#
# --system-site-packages is the load-bearing flag: the runpod image already
# carries a CUDA torch that satisfies the pyproject pin, and inheriting it is
# what keeps this from pulling 2.5GB off PyPI. Everything else the repo needs
# is small.
set -eux

python -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install -q -e /workspace/pinn

/workspace/venv/bin/python - <<'CHECK'
import torch

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
CHECK

/workspace/venv/bin/pinn --help >/dev/null

# setsid so it outlives the ssh session that installed it; one at a time.
pkill -f jobq-seppuku || true

if [ -s /usr/local/bin/jobq-seppuku ]; then
    chmod +x /usr/local/bin/jobq-seppuku
    setsid nohup /usr/local/bin/jobq-seppuku >/dev/null 2>&1 </dev/null &
    sleep 2
    # It refuses to arm without the injected pod id and key, and an unarmed
    # pod bills until someone notices.
    tail -1 /var/log/jobq-seppuku.log
fi
