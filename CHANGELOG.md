# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.1.0] - 2025-08-07
### Added
- `get_drive_file_metadata` method for retrieving file name and MIME type from Google Drive
- `list_files_in_folder` method for listing files in a Google Drive folder
- Progress bar for large file downloads using `tqdm`
- Column name ASCII normalization in DataFrame utilities
- Utility to remove unnamed columns automatically (e.g., from Excel/CSV exports)

### Changed
- Improved column name transformation to handle non-ASCII and non-string column names robustly

### Fixed
- More robust error handling in DataFrame utilities

## [1.0.0] - 2025-08-06
### Added
- Initial release of Google Sheets Helper
- Google API Client integration for reading Google Sheets and Excel files from Google Drive
- Data extraction to pandas DataFrame with automatic header detection
- DataFrame cleaning utilities: type optimization, missing value handling, text encoding cleanup, column naming transformation
- Flexible column naming: snake_case and camelCase support
- Comprehensive error handling with custom exception classes
- Logging setup utility
- Example usage scripts in `examples/`
- Full type hint support for all public APIs
- Documentation and API reference in README.md

