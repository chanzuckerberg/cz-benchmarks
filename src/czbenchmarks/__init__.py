import logging

logger = logging.getLogger(__name__)

def _enable_gpu_acceleration():
    try:
        import cuml.accel
        logger.info("Initializing GPU acceleration with cuML")
        cuml.accel.install()
        logger.info("GPU acceleration enabled successfully")
        return True
    except ImportError:
        logger.warning("cuML not found - GPU acceleration disabled")
        return False

HAS_GPU_ACCEL = _enable_gpu_acceleration()
