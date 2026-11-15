"""Complexity/coverage CRAP audit used when the named crapper tool is unavailable."""

import json
from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Class

ROOT = Path(__file__).resolve().parents[1]


def audit():
    coverage = json.loads((ROOT / "artifacts/coverage.json").read_text())["files"]
    records = []
    for path in sorted((ROOT / "src/adaptivetokstream").glob("*.py")):
        relative = str(path.relative_to(ROOT))
        data = coverage.get(relative, {})
        executed = set(data.get("executed_lines", []))
        missing = set(data.get("missing_lines", []))
        for block in cc_visit(path.read_text()):
            if isinstance(block, Class):
                continue
            lines = set(range(block.lineno, block.endline + 1))
            total = len((executed | missing) & lines)
            ratio = len(executed & lines) / total if total else 0.0
            complexity = block.complexity
            crap = complexity**2 * (1 - ratio) ** 3 + complexity
            records.append(
                {
                    "file": relative,
                    "function": block.fullname,
                    "line": block.lineno,
                    "complexity": complexity,
                    "coverage": ratio,
                    "crap": crap,
                }
            )
    result = {
        "formula": "CC^2*(1-coverage)^3+CC",
        "functions": records,
        "violations": [record for record in records if record["crap"] >= 10],
    }
    (ROOT / "artifacts/quality_audit.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "functions": len(records),
                "max_complexity": max(r["complexity"] for r in records),
                "max_crap": max(r["crap"] for r in records),
                "violations": result["violations"],
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    raise SystemExit(bool(audit()["violations"]))
