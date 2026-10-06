#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
results_root="${RESULTS_ROOT:-$repo_root/study_truth_transport_v2/results/remote_v3}"
report_root="$repo_root/study_truth_transport_v2/reporting/output"
figure_dir="$report_root/figures"
content="$report_root/report_content.json"
evidence="$report_root/evidence_inventory.json"
pdf="$repo_root/output/pdf/truth_transport_naacl_report.pdf"
pdf_python=${PDF_PYTHON:-python3}  # interpreter with reportlab, pypdf (and pypdfium2) installed

mkdir -p "$figure_dir" "$(dirname "$pdf")" /tmp/mpl-naacl-report

python3 "$repo_root/study_truth_transport_v2/reporting/validate_evidence.py" \
  --results-root "$results_root" \
  --output "$evidence"

MPLCONFIGDIR=/tmp/mpl-naacl-report MPLBACKEND=Agg python3 \
  "$repo_root/study_truth_transport_v2/reporting/make_figures.py" \
  --results-root "$results_root" \
  --output-dir "$figure_dir"

MPLCONFIGDIR=/tmp/mpl-naacl-report MPLBACKEND=Agg python3 \
  "$repo_root/study_truth_transport_v2/reporting/make_extension_figures.py" \
  --results-root "$results_root" \
  --output-dir "$figure_dir"

(cd "$repo_root" && python3 -m study_truth_transport_v2.reporting.compose_extended_report \
  --results-root "$results_root" \
  --figure-dir "$figure_dir" \
  --output "$content")

"$pdf_python" "$repo_root/study_truth_transport_v2/reporting/build_report.py" \
  --content "$content" \
  --output "$pdf"

"$pdf_python" - "$pdf" <<'PY'
import sys
from pathlib import Path
from pypdf import PdfReader

path = Path(sys.argv[1])
reader = PdfReader(str(path))
assert len(reader.pages) == 18, len(reader.pages)
assert path.stat().st_size > 100_000, path.stat().st_size
text = "\n".join(page.extract_text() or "" for page in reader.pages)
for required in (
    "Alignment is easy to raise",
    "RQ-A: shared facts raise cosine",
    "X3 and X6: external transfer",
    "NAACL assessment",
):
    assert required in text, required
print({"pdf": str(path.resolve()), "pages": len(reader.pages), "bytes": path.stat().st_size})
PY
