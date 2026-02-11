#!/usr/bin/env python3
"""Generate a self-contained HTML visualization from supervisor-worker JSONL logs.

Reads log files and embeds the data into index.html, producing a single
HTML file that can be opened directly in a browser.

Usage:
    python viz/visualize.py <log_dir> [-o <output.html>]
"""

import argparse
import json
import os
import sys


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def read_task_jsonl(path):
    """Read task.jsonl, returning {input: {...}, output: {...}}."""
    if not os.path.exists(path):
        return None
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "input":
                result["input"] = entry
            elif entry.get("type") == "output":
                result["output"] = entry
    return result or None


def main():
    p = argparse.ArgumentParser(description="Visualize supervisor-worker run logs")
    p.add_argument("log_dir", help="Directory containing JSONL log files")
    p.add_argument("--output", "-o", help="Output HTML path (default: <log_dir>/viz.html)")
    args = p.parse_args()

    data = {
        "supervisor": read_jsonl(os.path.join(args.log_dir, "supervisor.jsonl")),
        "worker": read_jsonl(os.path.join(args.log_dir, "worker.jsonl")),
        "repl": read_jsonl(os.path.join(args.log_dir, "repl.jsonl")),
        "task": read_task_jsonl(os.path.join(args.log_dir, "task.jsonl")),
    }

    if not data["supervisor"]:
        print(f"Error: No supervisor.jsonl found in {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    # Safely encode for embedding in <script> — escape HTML-breaking chars
    data_json = json.dumps(data, ensure_ascii=False)
    data_json = data_json.replace("<", "\\u003c").replace(">", "\\u003e")

    template_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(template_path) as f:
        template = f.read()

    html = template.replace("/*__DATA__*/null", data_json)

    output = args.output or os.path.join(args.log_dir, "viz.html")
    with open(output, "w") as f:
        f.write(html)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
