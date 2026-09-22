#!/usr/bin/env bash
# Install this app's dependencies into the conda env that is CURRENTLY ACTIVE.
#
#   conda create -n aitesting python=3.13 -y
#   conda activate aitesting
#   ./setup.sh
#
# Override the torch build if needed:  CUDA=cu126 ./setup.sh   (cpu for CPU-only)
set -e
cd "$(dirname "$0")"

CUDA="${CUDA:-cu130}"

if [ -z "$CONDA_PREFIX" ]; then
  echo "No conda env is active. Create and activate one first:"
  echo "  conda create -n aitesting python=3.13 -y && conda activate aitesting"
  exit 1
fi
echo "Installing into: $CONDA_DEFAULT_ENV ($CONDA_PREFIX)"
echo "Python: $(python -V)"

python -m pip install --upgrade pip

echo
echo "== torch ($CUDA) =="
if [ "$CUDA" = "cpu" ]; then
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
fi

echo
echo "== app requirements =="
python -m pip install -r requirements.txt

echo
echo "== check =="
python - <<'PY'
import cv2, numpy, torch, ultralytics, PySide6
print(f"PySide6      {PySide6.__version__}")
print(f"opencv       {cv2.__version__}")
print(f"ultralytics  {ultralytics.__version__}")
print(f"numpy        {numpy.__version__}")
print(f"torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu          {torch.cuda.get_device_name(0)}")
PY

echo
echo "Done. Edit config.json, then run:  python main.py"
