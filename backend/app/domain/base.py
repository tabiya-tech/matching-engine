"""Base class for the matching domain models."""

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base for every domain model: strict, closed, immutable values.

    ``strict=True`` turns off pydantic's type coercion (a ``"3"`` is not an ``int``), and
    ``extra="forbid"`` rejects fields the model does not declare. Upstream data is normalised into
    these models once, at the repository / file boundary (see ``app/domain/README.md``), so code
    past that boundary can rely on the declared types. ``frozen=True`` makes them values: derive a
    changed copy with ``model_copy(update=...)`` instead of assigning fields.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SourceRecord(BaseModel):
    """Base for models of documents produced by other systems (MongoDB, taxonomy files).

    These are the validation layer at the boundary, not domain models: they ignore fields the
    matching service does not read (upstream documents carry many more) and use the lenient
    coercing types in ``app.domain.coercion`` so a stray number-as-string does not reject a whole
    document.
    """

    model_config = ConfigDict(extra="ignore")
