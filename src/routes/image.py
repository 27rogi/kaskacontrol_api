import asyncio
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from src import state
from src.modules.inference import encode_jpeg, encode_jpeg_bytes

router = APIRouter(prefix="/api/image", tags=["image"])


async def read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(state.MAX_UPLOAD_BYTES + 1)
    if len(data) > state.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds upload limit")
    return data


@router.post("")
async def analyze(file: Annotated[UploadFile, File()], response: str = Query("json", pattern="^(json|image)$")):
    data = await read_upload(file)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    if state.service is None:
        raise HTTPException(status_code=503, detail="Inference model is not loaded")
    analysis = await asyncio.to_thread(state.service.analyze, frame)
    annotated = state.service.annotate(frame, analysis)
    if response == "image":
        return Response(content=encode_jpeg_bytes(annotated), media_type="image/jpeg")
    return {**analysis, "annotated_image": "data:image/jpeg;base64," + encode_jpeg(annotated)}
