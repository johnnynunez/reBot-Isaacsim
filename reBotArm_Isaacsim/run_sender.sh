#!/usr/bin/env bash
# 发送端启动脚本 / Sender launcher.
# 使用仓库根目录的 uv 工作空间环境运行 gravity_joint_sender.py。
# Run gravity_joint_sender.py inside the repo-root uv workspace environment.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -f "${UV_PYTHON}" ]]; then
  echo "[error] 未找到 uv 工作空间的 Python: ${UV_PYTHON} / uv workspace Python not found: ${UV_PYTHON}" >&2
  echo "[hint] 请先在仓库根目录 ${REPO_ROOT} 运行 uv sync / please run 'uv sync' at the repo root ${REPO_ROOT} first" >&2
  exit 1
fi

exec "${UV_PYTHON}" "${SCRIPT_DIR}/gravity_joint_sender.py" "$@"
