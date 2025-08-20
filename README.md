# Google Sheets Helper

A Python ETL driver for reading and transforming Google Sheets and Excel data from Google Drive. Simplifies the process of extracting spreadsheet data and converting it to database-ready pandas DataFrames with comprehensive optimization features.

[![PyPI version](https://img.shields.io/pypi/v/google-sheets-helper)](https://pypi.org/project/google-sheets-helper/)
[![Issues](https://img.shields.io/github/issues/machado000/google-sheets-helper)](https://github.com/machado000/google-sheets-helper/issues)
[![Last Commit](https://img.shields.io/github/last-commit/machado000/google-sheets-helper)](https://github.com/machado000/google-sheets-helper/commits/main)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/machado000/google-sheets-helper/blob/main/LICENSE)

## Features

- **Google Sheets & Excel Support**: Read Google Sheets and Excel files directly from Google Drive
- **Flexible Column Naming**: Choose between snake_case or camelCase column conventions, with robust ASCII normalization and automatic removal of unnamed columns (e.g., from Excel/CSV exports)
- **Progress Bar for Large Downloads**: Uses `tqdm` to show download progress for large Excel files
- **Advanced DataFrame type detection and cleaning**: Now available via the standalone package [`pandas-type-detector`](https://pypi.org/project/pandas-type-detector/)
- **Robust Error Handling**: Comprehensive error handling with specific exceptions
- **Progress Bar for Large Downloads**: Uses `tqdm` to show download progress for large Excel files
- **Type Hints**: Full type hint support for better IDE experience

## Installation

```bash
pip install google-sheets-helper
```

## Quick Start

### 1. Set up credentials

Place your Google service account credentials in `secrets/client_secret.json`.

### 2. Basic usage
setup_logging()

spreadsheet_id = "your_spreadsheet_id"
worksheet_name = "your_worksheet_name"

df = utils.fix_data_types(df, skip_columns=None)
df = utils.transform_column_names(df, naming_convention="snake_case")
```python
from google_sheets_helper import GoogleSheetsHelper, load_client_secret, setup_logging
from google_sheets_helper import DataframeUtils

setup_logging()
client_secret = load_client_secret()
gs_helper = GoogleSheetsHelper(client_secret)

spreadsheet_id = "your_spreadsheet_id"
worksheet_name = "your_worksheet_name"

df = gs_helper.load_sheet_as_dataframe(spreadsheet_id, worksheet_name)
utils = DataframeUtils()

df = utils.handle_missing_values(df)
df = utils.clean_text_encoding(df)
df = utils.transform_column_names(df, naming_convention="snake_case")

print(df.head(), df.dtypes)

print(df.head(), df.dtypes)
filename = os.path.join("data", f"{spreadsheet_id}_{worksheet_name}.csv")


```
os.makedirs("data", exist_ok=True)
filename = os.path.join("data", f"{spreadsheet_id}_{worksheet_name}.csv")

df.to_csv(filename, index=False)
```

## Data Cleaning Pipeline

You can use advanced DataFrame type detection and cleaning via the [`pandas-type-detector`](https://pypi.org/project/pandas-type-detector/) package:

```python
from google_sheets_helper import DataframeUtils

utils = DataframeUtils()
df = utils.handle_missing_values(df)
df = utils.clean_text_encoding(df)
df = utils.transform_column_names(df, naming_convention="snake_case")
df = utils.remove_unnamed_columns(df)
```

## API Reference

- `GoogleSheetsHelper`: Main class for reading and transforming Google Sheets/Excel data
- `get_drive_file_metadata`: Retrieve file name and MIME type from Google Drive
- `list_files_in_folder`: List files in a Google Drive folder
- `load_client_secret`: Loads credentials from a JSON file
- `setup_logging`: Configures logging for the package
- For advanced DataFrame cleaning and optimization, use [`pandas-type-detector`](https://pypi.org/project/pandas-type-detector/) and its `DataframeUtils` class
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
    df = gs_helper.load_sheet_as_dataframe(spreadsheet_id, worksheet_name)
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

- Python 3.9-3.12
- pandas >= 2.0.0
- gspread >= 5.10.0
- google-api-python-client >= 2.0.0
- tqdm >= 4.65.0

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
