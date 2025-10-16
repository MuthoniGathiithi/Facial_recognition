def __getattr__(name):
    if name == 'EnrollmentState':
        from .enrollment_state import EnrollmentState
        return EnrollmentState
    elif name == 'FaceEnrollment':
        from .enrollment_multiangle import FaceEnrollment
        return FaceEnrollment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['EnrollmentState', 'FaceEnrollment']