# Post-lock report errata

This directory contains delivery-time, presentation-only corrections discovered
after the immutable preregistration lock and formal aggregate analysis were
complete.  Files here are intentionally **not preregistered analysis code**.

`apply_report_presentation_erratum.py` verifies the original lock and frozen
analyzer byte-for-byte, patches only the missing report-sentence return in
memory, invokes the locked analyzer's `--report-only` path, and proves that all
non-report experiment artifacts remain unchanged.  It never parses the private
OOF prediction file.  Its immutable JSON manifests record each report render
and chain later commit/push-only delivery updates.

