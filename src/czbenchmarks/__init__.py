# Installing this at the root ensures that wherever sklearn is used in this library, it will be GPU accelerated by cuML if it's installed
def _enable_gpu_acceleration():
    try:
        import cuml.accel

        cuml.accel.install()
        return True
    except ImportError:
        return False


HAS_GPU_ACCEL = _enable_gpu_acceleration()
