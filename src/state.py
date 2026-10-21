from concurrent.futures import ThreadPoolExecutor

from src.config import MAX_UPLOAD_BYTES as MAX_UPLOAD_BYTES
from src.config import UPLOAD_DIRECTORY, VIDEO_WORKERS
from src.config import VIDEO_FRAME_STEP as VIDEO_FRAME_STEP
from src.modules.inference import InferenceService
from src.modules.rtsp_manager import RTSPMonitorManager

UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
executor = ThreadPoolExecutor(max_workers=VIDEO_WORKERS)
service: InferenceService | None = None
rtsp_manager: RTSPMonitorManager | None = None
