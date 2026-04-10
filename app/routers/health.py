from typing import Annotated, Optional

import torch
from fastapi import APIRouter, Depends

from config import Config
from dependencies import get_config
from models.healthcheck_response import GpuInfo, HealthCheckResponse

router = APIRouter()

@router.get("/health")
async def health(config = Annotated[Config, Depends(get_config)]) -> HealthCheckResponse:
    gpu_info: Optional[GpuInfo] = None
    if config.model_config.device:
        props = torch.cuda.get_device_properties(0)
        gpu_info = GpuInfo(
            name = torch.cuda.get_device_name(0),
            memory_total_gb = round(props.total_memory / 1e9, 2),
            memory_used_gb = round(torch.cuda.memory_allocated(0) / 1e9, 2),
            memory_free_gb = round((props.total_memory - torch.cuda.memory_allocated(0)) / 1e9, 2),
        )
    return HealthCheckResponse(
        healthy = True,
        model = config.model_config.model_size,
        device = config.model_config.device.upper(),
        fp16 = config.model_config.use_fp16,
        cuda_available = torch.cuda.is_available(),
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None,
        gpu_info = gpu_info,
    )
