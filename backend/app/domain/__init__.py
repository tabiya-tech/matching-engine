"""Typed domain models of the matching service (jobs, occupations, users, scores).

Import from the submodules (``app.domain.job``, ``app.domain.user``, …); this package does not
re-export them, which keeps ``app.schemas`` ↔ ``app.domain`` imports acyclic. The field-by-field
data dictionary, with each field's upstream source, is ``app/domain/README.md``.
"""
