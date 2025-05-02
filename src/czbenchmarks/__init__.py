import logging

logger = logging.getLogger(__name__)

def _enable_gpu_acceleration():
    try:
        import cuml.accel
        cuml.accel.install()
        logger.info("Initialized GPU acceleration with cuML")
        return True
    except ImportError:
        logger.warning("cuML not found - GPU acceleration disabled")
        return False

HAS_GPU_ACCEL = _enable_gpu_acceleration()
