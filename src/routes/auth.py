from fastapi import APIRouter, HTTPException

from src.auth import ADMIN_USERNAME, token, valid_login
from src.schemas.auth import LoginRequest

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login")
async def login(payload: LoginRequest):
    if not valid_login(payload):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"token": token(), "username": ADMIN_USERNAME}
