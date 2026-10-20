#!/usr/bin/env python3
"""Build an occupation file (``OccupationFileEntry`` JSON list) from a Tabiya taxonomy export.

The output is what ``scripts/ingest_occupations.py`` loads into a deployment's database. One entry
per ISCO-08 unit group (4-digit ``iscogroup``), with the union of its occupations' essential and
optional skills and the export's labels and descriptions (so a Spanish export gives a Spanish
catalogue). See ``app.occupations.taxonomy_export`` for the rules.

The export has no county attributes, expected demand, O*NET work activities, post-secondary flag or
task list:

* ``--enrich-from`` copies work activities, the post-secondary flag and the task list from an
  existing occupation file, by ISCO code (``0110`` matches ``110``). They are language-neutral.
* ``--merge-skills`` also adds ``--enrich-from``'s skills (some are curated and in no taxonomy
  relation: without it 10 unit groups of the Argentina export have no skills). They are labelled
  from the export, so the catalogue stays in the export's language.
* ``--counties`` gives each occupation one ``counties_data`` row per name, with no estimates.
  ``/match_v4`` keeps only occupations in the user's county, so an occupation with no county
  rows is never recommended: list the deployment's regions here.
* ``--copy-counties`` copies ``counties_data`` (estimates and demand) from ``--enrich-from``
  instead, e.g. to rebuild a Kenya file from a newer taxonomy.

Usage (from ``backend/``)::

    python -m scripts.build_occupation_file \\
        --taxonomy-dir ../../empujar/empujar-taxonomy \\
        --enrich-from resources/occupations/en/combined_occupation_database_with_wa.json \\
        --merge-skills \\
        --counties "Buenos Aires,CABA,Córdoba,Santa Fe,Mendoza" \\
        --output resources/occupations/es/occupations_argentina.json

    # report only
    python -m scripts.build_occupation_file --taxonomy-dir ... --dry-run

Writes ``<output>`` and ``<output minus .json>.meta.json`` (taxonomy model, counts, arguments), which
the ingest records as the documents' source.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.occupations.ingest import (  # noqa: E402
    OccupationFileInvalid,
    parse_occupation_file,
)
from app.occupations.taxonomy_export import (  # noqa: E402
    OccupationBuildReport,
    build_occupation_entries,
    read_taxonomy_export,
)

_SHOW = 10


def meta_path(output: Path) -> Path:
    """Sidecar of an occupation file: ``foo.json`` -> ``foo.meta.json``."""
    return output.with_suffix(".meta.json")


def _print_report(report: OccupationBuildReport, n_counties: int) -> None:
    def codes(label: str, values: list[str]) -> None:
        if values:
            more = f" (+{len(values) - _SHOW} more)" if len(values) > _SHOW else ""
            print(f"  {label}: {len(values)} — {', '.join(values[:_SHOW])}{more}")

    print(f"  unit groups built: {report.n_unit_groups}")
    print(f"  counties per occupation: {n_counties or 'none'}")
    codes("without skills", report.codes_without_skills)
    codes("without description", report.codes_without_description)
    codes("not in --enrich-from", report.codes_not_enriched)
    if report.n_enrichment_unused:
        print(
            f"  --enrich-from entries matching no unit group: {report.n_enrichment_unused}"
        )
    if report.n_occupations_outside_unit_groups:
        print(
            "  occupations under no unit group (skills unused): "
            f"{report.n_occupations_outside_unit_groups}"
        )
    if report.skill_diffs:
        print(
            f"  taxonomy skills differ from --enrich-from for {len(report.skill_diffs)} entries "
            "(missing/extra essential, missing/extra optional; before --merge-skills):"
        )
        for d in report.skill_diffs[:_SHOW]:
            print(
                f"    {d.code}: -{d.missing_essential}/+{d.extra_essential} essential, "
                f"-{d.missing_optional}/+{d.extra_optional} optional"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build an occupation file from a taxonomy export."
    )
    ap.add_argument(
        "--taxonomy-dir",
        required=True,
        type=Path,
        help="Taxonomy export directory (occupations.csv, occupation_groups.csv, …).",
    )
    ap.add_argument(
        "--enrich-from",
        type=Path,
        help="Existing occupation file to copy work activities, post-secondary flag and tasks from.",
    )
    ap.add_argument(
        "--merge-skills",
        action="store_true",
        help="Add --enrich-from's skills to the built ones (labelled from the export).",
    )
    counties = ap.add_mutually_exclusive_group()
    counties.add_argument(
        "--counties",
        default="",
        help="Comma-separated county / province names; one empty counties_data row each.",
    )
    counties.add_argument(
        "--copy-counties",
        action="store_true",
        help="Copy counties_data (estimates, demand) from --enrich-from.",
    )
    ap.add_argument("--output", type=Path, help="Occupation file to write.")
    ap.add_argument(
        "--dry-run", action="store_true", help="Report only; write nothing."
    )
    args = ap.parse_args(argv)

    if not args.dry_run and args.output is None:
        ap.error("--output is required unless --dry-run")
    if (args.copy_counties or args.merge_skills) and args.enrich_from is None:
        ap.error("--copy-counties and --merge-skills need --enrich-from")

    enrich = []
    if args.enrich_from is not None:
        try:
            enrich = parse_occupation_file(
                json.loads(args.enrich_from.read_text(encoding="utf-8"))
            )
        except OccupationFileInvalid as e:
            print(
                f"[build_occupation_file] --enrich-from is invalid: {e}",
                file=sys.stderr,
            )
            for problem in e.problems[:_SHOW]:
                print(f"  {problem}", file=sys.stderr)
            return 2
    county_names = [c.strip() for c in args.counties.split(",") if c.strip()]

    export = read_taxonomy_export(args.taxonomy_dir)
    entries, report = build_occupation_entries(
        export,
        enrich_from=enrich,
        counties=county_names,
        copy_counties=args.copy_counties,
        merge_skills=args.merge_skills,
    )
    # The output must pass the same validation the ingest applies.
    parse_occupation_file(entries)

    model = export.model_info
    print(
        f"[build_occupation_file] {args.taxonomy_dir} "
        f"({model.name + ' ' + model.locale if model else 'no model_info.csv'})"
    )
    _print_report(report, len(county_names))
    if not county_names and not args.copy_counties:
        print(
            "  WARNING: no counties — /match_v4 recommends no occupation without a county row "
            "(pass --counties)."
        )

    if args.dry_run:
        print("[build_occupation_file] dry run: nothing written")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    meta = {
        "taxonomy_model": model.model_dump() if model else None,
        "taxonomy_dir": str(args.taxonomy_dir),
        "enrich_from": str(args.enrich_from) if args.enrich_from else None,
        "counties": county_names,
        "copy_counties": args.copy_counties,
        "merge_skills": args.merge_skills,
        "n_occupations": len(entries),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path(args.output).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[build_occupation_file] wrote {len(entries)} occupations to {args.output} "
        f"and {meta_path(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
