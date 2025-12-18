class RekoError(Exception):
    exit_code = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class InputError(RekoError):
    exit_code = 2


class OutputError(RekoError):
    pass


class ProcessingError(RekoError):
    pass


class ExternalServiceError(RekoError):
    pass


class YouTubeError(ExternalServiceError):
    pass


class TranscriptError(ExternalServiceError):
    pass
