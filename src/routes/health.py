from fastapi import APIRouter

from src import state

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    return {
        "status": "ok" if state.service else "loading",
        "model": state.service.model_name if state.service else None,
    }
