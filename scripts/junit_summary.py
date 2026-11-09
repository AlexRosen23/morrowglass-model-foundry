"""Summarise a pytest JUnit XML file into reports/test_summary.json (counts only; evidence for metrics.json)."""

import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
import subprocess

root = ET.parse(sys.argv[1]).getroot()
suite = root if root.tag == "testsuite" else root.find("testsuite")
assert suite is not None
total, failures, errors, skipped = (int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
out = {"total": total, "passed": total - failures - errors - skipped, "failed": failures, "errors": errors,
       "skipped": skipped, "command": "make test", "commit": commit, "time": datetime.now(UTC).isoformat(timespec="seconds"),
       "platform": sys.platform}
Path("reports/test_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(out)
