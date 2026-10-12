"""O*NET work activities of a job or occupation."""

import logging
from collections.abc import Iterable

from pydantic import Field

from app.domain.base import DomainModel, SourceRecord
from app.domain.coercion import OptionalNumber, OptionalText

logger = logging.getLogger(__name__)


class WorkActivity(DomainModel):
    """One O*NET work activity of a job or occupation, with its importance and level.

    Scored against the user's ``PreferenceVector.bws_scores`` (keyed by the same code). Serialises
    under the upstream ``WA_*`` names (``model_dump(by_alias=True)``), which the ranking engine reads.
    """

    code: str = Field(
        serialization_alias="WA_code",
        description="O*NET work-activity element id, e.g. ``4.A.4.a.4``.",
    )
    label: str = Field(
        serialization_alias="WA_label",
        description="O*NET work-activity name; empty when the source has none.",
    )
    importance: float = Field(
        serialization_alias="WA_Importance",
        description="O*NET importance of the activity for the occupation, on the 1–5 scale.",
    )
    level: float = Field(
        serialization_alias="WA_Level",
        description="O*NET level of the activity for the occupation, on the 1–7 scale.",
    )


class WorkActivityRecord(SourceRecord):
    """A work activity as stored by the llm-reranker (job documents) and the occupation file.

    ``WA_Importance`` / ``WA_Level`` arrive as numbers, numeric strings, ``""`` (not rated) or the
    llm-reranker's ``"code is not available"``; non-numeric values parse to None.
    """

    WA_code: OptionalText = Field(
        default=None, description="O*NET work-activity element id."
    )
    WA_label: OptionalText = Field(
        default=None, description="O*NET work-activity name."
    )
    WA_Importance: OptionalNumber = Field(
        default=None, description="Importance (1–5), when rated."
    )
    WA_Level: OptionalNumber = Field(
        default=None, description="Level (1–7), when rated."
    )


def work_activities_from_records(
    records: Iterable[WorkActivityRecord], *, source_id: str
) -> list[WorkActivity]:
    """Rated work activities from stored records.

    Records without a code, or without a numeric importance and level, are dropped: the engine
    cannot weight an unrated activity.

    :param records: The stored work-activity records
    :param source_id: Id of the job or occupation, for the log line
    :return: The rated work activities, in source order
    """
    activities: list[WorkActivity] = []
    dropped = 0
    for record in records:
        if (
            not record.WA_code
            or record.WA_Importance is None
            or record.WA_Level is None
        ):
            dropped += 1
            continue
        activities.append(
            WorkActivity(
                code=record.WA_code,
                label=record.WA_label or "",
                importance=record.WA_Importance,
                level=record.WA_Level,
            )
        )
    if dropped:
        logger.debug("%s: dropped %d unrated work activities", source_id, dropped)
    return activities
