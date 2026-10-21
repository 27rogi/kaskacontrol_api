import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src import state
from src.auth import middleware
from src.config import CORS_ORIGIN_REGEX, CORS_ORIGINS, HOST, PORT, RELOAD
from src.modules.db.database import init_db
from src.modules.inference import load_service
from src.modules.rtsp_manager import RTSPMonitorManager
from src.modules.snapshots import VIOLATION_IMAGE_DIR
from src.routes.analytics import router as analytics_router
from src.routes.auth import router as auth_router
from src.routes.health import router as health_router
from src.routes.image import router as image_router
from src.routes.monitors import router as monitors_router
from src.routes.videos import router as videos_router
from src.routes.websocket import router as websocket_router
from src.routes.zones import router as zones_router


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    init_db()
    state.service = await asyncio.to_thread(load_service)
    state.rtsp_manager = RTSPMonitorManager(state.service)
    await state.rtsp_manager.start_enabled()
    yield
    await state.rtsp_manager.shutdown()
    state.service = None


app = FastAPI(title="Helmet Detection API", version="1.0.0", lifespan=lifespan)
app.mount("/uploads/violations", StaticFiles(directory=VIOLATION_IMAGE_DIR), name="violation-images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(middleware)
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(zones_router)
app.include_router(monitors_router)
app.include_router(analytics_router)
app.include_router(image_router)
app.include_router(videos_router)
app.include_router(websocket_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_config=None,
    )
