#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_env.py
Usage:
    python export_env.py --out ./env_snapshot
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import shutil

def run(cmd):
    """Return stdout of a shell command or '' if it fails."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"✔ Saved {path}")

def dump_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"✔ Saved {path}")

def collect_system_info():
    info = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }

    # GPU 信息（若安装了 torch）
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    return info

def export_pip_requirements(out_dir: Path):
    reqs = run([sys.executable, "-m", "pip", "freeze"])
    save_text(out_dir / "requirements.txt", reqs or "# pip freeze 无输出")

def export_conda_yaml(out_dir: Path):
    if shutil.which("conda") and os.environ.get("CONDA_DEFAULT_ENV"):
        yaml_text = run(["conda", "env", "export", "--no-builds"])
        if yaml_text:
            save_text(out_dir / "conda_env.yaml", yaml_text)
    else:
        print("ℹ 未检测到 Conda 环境，跳过 conda_env.yaml")

def export_env_vars(out_dir: Path):
    dump_json(out_dir / "env_vars.json", dict(os.environ))

def main():
    parser = argparse.ArgumentParser(description="Export current environment configuration")
    parser.add_argument("--out", "-o", default="env_snapshot", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 系统信息
    dump_json(out_dir / "system_info.json", collect_system_info())

    # 2) pip requirements
    export_pip_requirements(out_dir)

    # 3) conda YAML（若存在）
    export_conda_yaml(out_dir)

    # 4) 环境变量
    export_env_vars(out_dir)

    print("\n✅ 环境导出完成！全部文件位于：", out_dir)

if __name__ == "__main__":
    main()
