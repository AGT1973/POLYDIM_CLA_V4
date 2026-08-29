from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    env = os.environ.copy()
    env["POLYDIM_PROJECT"] = str(project)
    cmd = [sys.executable, "-m", "pytest", "scientific_suite/tests", "-q", "--tb=short", "--junitxml=scientific_suite/report.xml"]
    proc = subprocess.run(cmd, cwd=project.parent if project.name == "scientific_suite" else project, env=env, text=True)
    result = {"project": str(project), "returncode": proc.returncode}
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    raise SystemExit(proc.returncode)

if __name__ == "__main__":
    main()
