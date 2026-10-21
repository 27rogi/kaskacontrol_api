from typing import Any

HELMET_COLOR_ROLES = {
    "white": "Руководство",
    "red": "Прорабы / мастера",
    "yellow": "Рабочие",
    "orange": "Рабочие",
    "unknown": "Не определена",
}

VISUAL_ROLES = {
    "white": "management",
    "red": "foreman/master",
    "yellow": "worker",
    "orange": "worker",
    "unknown": "unknown role",
}


def role_for_color(color: str | None) -> str:
    return HELMET_COLOR_ROLES.get((color or "unknown").lower(), HELMET_COLOR_ROLES["unknown"])


def visual_role_for_color(color: str | None) -> str:
    return VISUAL_ROLES.get((color or "unknown").lower(), VISUAL_ROLES["unknown"])


def role_config() -> dict[str, Any]:
    return {"colors": HELMET_COLOR_ROLES.copy()}
