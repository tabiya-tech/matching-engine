import asyncio
from typing import List, Union

from fastapi import APIRouter, HTTPException

from app.schemas import MatchRequest, MatchResponse
from app.services.matching_service import match_single_user


router = APIRouter()


@router.post(
    "/match",
    tags=["matching"],
    response_model=List[MatchResponse],
    responses={
        400: {
            "description": "Bad Request - invalid payload content",
            "content": {"application/json": {"example": {"detail": "user must include user_id"}}},
        },
        500: {
            "description": "Internal Server Error",
            "content": {"application/json": {"example": {"detail": "Internal server error"}}},
        },
    },
)
async def match(payload: List[MatchRequest]):
    """Match one user or many users (batch).

    - If you send a single user object, you get a single MatchResponse.
    - If you send an array of user objects, you get a list of MatchResponse.
    """

    try:
        if isinstance(payload, list):
            # Process batch concurrently for better performance
            tasks = [match_single_user(user.model_dump()) for user in payload]
            return await asyncio.gather(*tasks)

        return await match_single_user(payload.model_dump())
    except ValueError as e:
        # Matching service uses ValueError for invalid inputs
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e.__class__.__name__}")
