"""Liveness route."""

from fastapi import APIRouter
from pydantic import BaseModel


class Health(BaseModel):
    status: str


def add_health_routes(router: APIRouter) -> None:
    """
    Adds ``GET /health`` to the router.

    :param router: The authenticated service router
    """

    @router.get("/health")
    async def health() -> Health:
        return Health(status="ok")
