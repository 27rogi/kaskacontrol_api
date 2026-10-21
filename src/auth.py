import hashlib
import hmac
from secrets import compare_digest

from fastapi import Request, Response

from src.config import ADMIN_PASSWORD, ADMIN_USERNAME
from src.schemas.auth import LoginRequest

AUTH_SECRET = "ryzhenkov_igor_ybvt2304"


def equal(left: str, right: str) -> bool:
    return compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def token() -> str:
    return hmac.new(AUTH_SECRET.encode(), f"{ADMIN_USERNAME}:{ADMIN_PASSWORD}".encode(), hashlib.sha256).hexdigest()


def authorized(request: Request) -> bool:
    value = request.headers.get("authorization", "")
    return value.startswith("Bearer ") and equal(value[7:], token())


async def middleware(request: Request, call_next):
    path = request.url.path
    public = path in {"/api/auth/login", "/api/health"} or path.startswith(("/docs", "/openapi.json"))
    if request.method == "OPTIONS" or public or not path.startswith("/api/") or authorized(request):
        return await call_next(request)
    return Response('{"detail":"Authentication required"}', status_code=401, media_type="application/json")


def valid_login(payload: LoginRequest) -> bool:
    return equal(payload.username, ADMIN_USERNAME) and equal(payload.password, ADMIN_PASSWORD)
