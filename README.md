# Google Sheets Helper

A Python ETL driver for reading and transforming Google Sheets and Excel data from Google Drive. Simplifies the process of extracting spreadsheet data and converting it to database-ready Python lists of dictionaries with comprehensive optimization features.

[![PyPI version](https://img.shields.io/pypi/v/google-sheets-helper)](https://pypi.org/project/google-sheets-helper/)
[![Issues](https://img.shields.io/github/issues/machado000/google-sheets-helper)](https://github.com/machado000/google-sheets-helper/issues)
[![Last Commit](https://img.shields.io/github/last-commit/machado000/google-sheets-helper)](https://github.com/machado000/google-sheets-helper/commits/main)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/machado000/google-sheets-helper/blob/main/LICENSE)

## Features

- **Google Sheets & Excel Support**: Read Google Sheets and Excel files directly from Google Drive
- **Flexible Column Naming**: Choose between snake_case or camelCase column conventions, with robust ASCII normalization and automatic removal of unnamed columns (e.g., from Excel/CSV exports)
- **Progress Logging for Large Downloads**: Shows download progress for large Excel files
- **Pure Python Data Cleaning and Transformation**: No pandas required; all utilities work with list-of-dictionaries
- **Robust Error Handling**: Comprehensive error handling with specific exceptions
- **Type Hints**: Full type hint support for better IDE experience

## Installation

```bash
pip install google-sheets-helper
```

## Quick Start

### 1. Set up credentials

Place your Google service account credentials in `secrets/client_secret.json`.

### 2. Basic usage

```python
from google_sheets_helper import GoogleSheetsHelper, WorksheetUtils, load_client_secret

client_secret = load_client_secret()
gs_helper = GoogleSheetsHelper(client_secret)

spreadsheet_id = "your_spreadsheet_id"
worksheet_name = "your_worksheet_name"

# Load data as list of dictionaries
data = gs_helper.load_sheet_as_json(spreadsheet_id, worksheet_name)

utils = WorksheetUtils()
data = utils.handle_missing_values(data)
data = utils.clean_text_encoding(data)
data = utils.transform_column_names(data, naming_convention="snake_case")

# Print first row
print(data[0])

# Save to CSV
import csv
with open("output.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
```

## Data Cleaning Pipeline

All data cleaning and transformation utilities now work with list-of-dictionaries:

```python
from google_sheets_helper import WorksheetUtils

utils = WorksheetUtils()
data = utils.handle_missing_values(data)
data = utils.clean_text_encoding(data)
data = utils.transform_column_names(data, naming_convention="snake_case")
data = utils.remove_unnamed_and_null_columns(data)
```

## API Reference

- `GoogleSheetsHelper`: Main class for reading and transforming Google Sheets/Excel data
- `get_drive_file_metadata`: Retrieve file name and MIME type from Google Drive
- `list_files_in_folder`: List files in a Google Drive folder
- `load_client_secret`: Loads credentials from a JSON file
- `WorksheetUtils`: Pure Python utilities for cleaning and transforming list-of-dictionaries
- Exception classes: `AuthenticationError`, `APIError`, `ConfigurationError`, `DataProcessingError`, `ValidationError`

## Error Handling

```python
from google_sheets_helper import (
    GoogleSheetsHelper,
    AuthenticationError,
    ValidationError,
    APIError,
    DataProcessingError,
    ConfigurationError
)

try:
    data = gs_helper.load_sheet_as_json(spreadsheet_id, worksheet_name)
except AuthenticationError:
    # Handle credential issues
    pass
except ValidationError:
    # Handle input validation errors
    pass
except APIError:
    # Handle API errors
    pass
except DataProcessingError:
    # Handle data processing errors
    pass
```

## Examples

Check the `examples/` directory for comprehensive usage examples:

- `basic_usage.py` - Simple sheet extraction and cleaning

## Requirements

- Python 3.10-3.12
- gspread >= 6.0.0
- google-api-python-client >= 2.0.0
- python-calamine >= 0.4.0

## Development

For development installation:

```bash
git clone https://github.com/machado000/google-sheets-helper
cd google-sheets-helper
pip install -e ".[dev]"
```

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Support

- [Documentation](https://github.com/machado000/google-sheets-helper#readme)
- [Issues](https://github.com/machado000/google-sheets-helper/issues)
- [Examples](examples/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
