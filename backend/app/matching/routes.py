"""Route for matching users: ``POST /match``."""

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.config import (
    MATCH_TOP_K_SKILL_GAPS,
    MATCH_V2_MAX_USERS_PER_REQUEST,
    MATCH_V4_FINAL_TOP_K,
    MATCH_V4_RETRIEVE_TOP_K,
)
from app.constants.errors import (
    FORBIDDEN_RESPONSE_DOC,
    INTERNAL_ERROR_RESPONSE_DOC,
    error_response_doc,
)
from app.matching.errors import InvalidMatchRequestError
from app.matching.examples import MATCH_BODY_DESCRIPTION, MATCH_BODY_EXAMPLE
from app.matching.get_matching_service import get_matching_service
from app.matching.service import IMatchingService
from app.matching.types import MatchOptions
from app.schemas import MatchRequest, MatchResponse

logger = logging.getLogger(__name__)


def add_matching_routes(router: APIRouter) -> None:
    """
    Adds the matching routes to the router.

    :param router: The authenticated service router
    """

    @router.post(
        "/match",
        tags=["matching"],
        operation_id="match",
        summary="Match users to occupations, job opportunities and skill gaps",
        response_model=list[MatchResponse],
        responses={
            400: error_response_doc(
                "Bad Request: empty body, too many users, invalid "
                "``final_score_combiner``, or an input the engine rejects.",
                "Request body must be a non-empty JSON array.",
            ),
            403: FORBIDDEN_RESPONSE_DOC,
            422: error_response_doc(
                "Unprocessable Entity: the body is not an array, a field has the wrong type, "
                "or a query parameter is out of range.",
                "body.0.skills_vector: Field required",
            ),
            500: INTERNAL_ERROR_RESPONSE_DOC,
        },
    )
    async def match(
        payload: Annotated[
            list[MatchRequest],
            Body(
                ...,
                description=(
                    MATCH_BODY_DESCRIPTION
                    + f" At most {MATCH_V2_MAX_USERS_PER_REQUEST} users per request."
                ),
                example=MATCH_BODY_EXAMPLE,
            ),
        ],
        retrieve_top_k: int | None = Query(
            None,
            ge=1,
            le=500,
            description=(
                "Stage-1 embedding-cosine shortlist size per user, before cross-encoder rerank. "
                f"Default: {MATCH_V4_RETRIEVE_TOP_K}."
            ),
        ),
        final_top_k: int | None = Query(
            None,
            ge=1,
            le=200,
            description=(
                "Cross-encoder pool size and maximum number of opportunities returned per user. "
                f"Default: {MATCH_V4_FINAL_TOP_K}."
            ),
        ),
        final_score_combiner: str | None = Query(
            None,
            description=(
                "How ``final_score`` combines ``u_hat`` and ``p_hat``: ``product`` (u_hat × p_hat) or "
                "``geometric_mean`` (√(u_hat × p_hat)). Defaults to the server's FINAL_SCORE_COMBINER "
                "(``product`` unless overridden)."
            ),
        ),
        skill_gap_top_k: int | None = Query(
            None,
            ge=1,
            le=50,
            description=f"Number of skill-gap recommendations per user. Default: {MATCH_TOP_K_SKILL_GAPS}.",
        ),
        matching_service: IMatchingService = Depends(get_matching_service),
    ) -> list[MatchResponse]:
        """Match one or more users to occupations, job opportunities and skill gaps.

        Body is a JSON array of ``MatchRequest`` (use length 1 for a single user); the response is a
        list of ``MatchResponse`` in the same order. Candidates are shortlisted by skill similarity,
        re-scored by a cross-encoder, and ranked by ``final_score`` (preference utility × success
        propensity, adjusted for skill coverage and location); each factor is in ``score_breakdown``.

        Requires the ``x-api-key`` header.
        """
        try:
            out = await matching_service.match(
                payload,
                MatchOptions(
                    retrieve_top_k=retrieve_top_k,
                    final_top_k=final_top_k,
                    final_score_combiner=final_score_combiner,
                    skill_gap_top_k=skill_gap_top_k,
                ),
            )
            return out

        except InvalidMatchRequestError as e:
            logger.warning(str(e))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except ValueError as e:
            logger.exception(e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception(e)
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
            ) from e
