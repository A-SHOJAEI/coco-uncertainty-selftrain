#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -x "${VENV_DIR}/bin/python" && -x "${VENV_DIR}/bin/pip" ]]; then
  exit 0
fi

rm -rf "${VENV_DIR}"

# Do not rely on ensurepip: create venv without pip, then bootstrap pip via get-pip.py.
"${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"

GETPIP="${VENV_DIR}/get-pip.py"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -o "${GETPIP}" https://bootstrap.pypa.io/get-pip.py
elif command -v wget >/dev/null 2>&1; then
  wget -qO "${GETPIP}" https://bootstrap.pypa.io/get-pip.py
else
  # Fall back to Python-only download (requests is not guaranteed; use stdlib).
  VENV_DIR="${VENV_DIR}" "${VENV_DIR}/bin/python" - <<'PY'
import os
import urllib.request
url = "https://bootstrap.pypa.io/get-pip.py"
venv = os.environ["VENV_DIR"]
out = os.path.join(venv, "get-pip.py")
urllib.request.urlretrieve(url, out)
print("Downloaded", url, "->", out)
PY
fi

"${VENV_DIR}/bin/python" "${GETPIP}" --disable-pip-version-check
rm -f "${GETPIP}"

# Hard fail if pip isn't available.
"${VENV_DIR}/bin/pip" --version >/dev/null
