"""
Custom exceptions for the Google Sheets Helper module.
"""


class GoogleSheetsHelperError(Exception):
    """Base exception for all Google Sheets Helper errors."""

    def __init__(self, message: str, original_error=None, **context):
        self.message = message
        self.original_error = original_error
        self.context = context
        if original_error:
            super().__init__(f"{message} (caused by: {str(original_error)})")
        else:
            super().__init__(message)


class AuthenticationError(GoogleSheetsHelperError):
    """Raised when authentication with Google Sheets API fails."""
    pass


class APIError(GoogleSheetsHelperError):
    """Raised when Google Ads API returns an error."""
    pass


class ConfigurationError(GoogleSheetsHelperError):
    """Raised when configuration is invalid."""
    pass


class DataProcessingError(GoogleSheetsHelperError):
    """Raised when data processing/transformation fails."""
    pass


class ValidationError(GoogleSheetsHelperError):
    """Raised when input validation fails."""
    pass
