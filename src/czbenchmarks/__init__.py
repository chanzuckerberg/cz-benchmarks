def _enable_gpu_acceleration():
    try:
        import cuml.accel
        cuml.accel.install()
        return True
    except ImportError:
        return False

HAS_GPU_ACCEL = _enable_gpu_acceleration()
