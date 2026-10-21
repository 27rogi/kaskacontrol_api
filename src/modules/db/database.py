import threading

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from src.config import DATABASE_FILE

# для SQLAlchemy надо импортировать, иначе не создаются таблицы
from src.schemas.base import DatabaseBase as Base
from src.schemas.violation import ViolationEvent  # noqa: F401
from src.schemas.zone import Zone  # noqa: F401

DATABASE_URL = f"sqlite:///{DATABASE_FILE.as_posix()}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(engine, expire_on_commit=False)
db_lock = threading.RLock()


def init_db() -> None:
    Base.metadata.create_all(engine)
    if "violation_events" in inspect(engine).get_table_names():
        columns = {column["name"] for column in inspect(engine).get_columns("violation_events")}
        with engine.begin() as connection:
            for column in ("start_image_path", "end_image_path", "helmet_role"):
                if column not in columns:
                    default = " DEFAULT 'Не определена'" if column == "helmet_role" else ""
                    connection.execute(text(f"ALTER TABLE violation_events ADD COLUMN {column} TEXT{default}"))
