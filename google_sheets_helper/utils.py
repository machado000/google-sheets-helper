"""
Utility functions manipulate list of dictionaries and setup logging.
"""
import json
import logging
import os
import re

from typing import Any, Optional
from unicodedata import normalize, combining
from .exceptions import ConfigurationError


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_client_secret(client_secret_path: Optional[str] = None) -> dict[str, Any]:
    """
    Load Google Ads API credentials from JSON file.

    Args:
        client_secret_path (Optional[str]): Path to the credentials file. If None, tries default locations.

    Returns:
        dict[str, Any]: Loaded client_secret.json credentials.

    Raises:
        FileNotFoundError: If credentials file is not found.
        json.JSONDecodeError: If JSON parsing fails.
    """
    default_paths = [
        os.path.join("secrets", "client_secret.json"),
        os.path.join(os.path.expanduser("~"), ".client_secret.json"),
        "client_secret.json"
    ]

    if client_secret_path:
        paths_to_try = [client_secret_path]
    else:
        paths_to_try = default_paths

    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    credentials = json.load(f)

                if not credentials:
                    raise ConfigurationError(f"Credentials file {path} is empty")

                if not isinstance(credentials, dict):
                    raise ConfigurationError(f"Credentials file {path} must contain a JSON dictionary")

                return credentials

            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON file {path}: {e}")
                raise ConfigurationError(
                    f"Invalid JSON format in credentials file {path}",
                    original_error=e
                ) from e

            except IOError as e:
                raise ConfigurationError(
                    f"Failed to read credentials file {path}",
                    original_error=e
                ) from e

    raise ConfigurationError(
        f"Could not find credentials file in any of these locations: {paths_to_try}"
    )


class WorksheetUtils:
    """
    Utility class for list of dictionaries operations with enhanced data type detection and cleaning.
    This replaces pandas DataFrame operations with pure Python equivalents.

    Example usage:
        utils = WorksheetUtils()
        data = utils.clean_text_encoding(data)
        data = utils.handle_missing_values(data)
        data = utils.transform_column_names(data, naming_convention="snake_case")
    """

    def __init__(self):
        """
        Initialize WorksheetUtils.
        """

    @staticmethod
    def clean_text_encoding(data: list[dict[str, Any]],
                            max_length: int = 255,
                            normalize_whitespace: bool = True) -> list[dict[str, Any]]:
        """
        Enhanced text cleaning with configurable options.

        Args:
            data (list[dict[str, Any]]): Input list of dictionaries.
            max_length (int): Maximum length for text fields.
            normalize_whitespace (bool): Whether to normalize whitespace.

        Returns:
            list[dict[str, Any]]: List of dictionaries with cleaned text values (copy).
        """
        if not data:
            return data.copy()

        # Create a deep copy to avoid modifying the original
        cleaned_data = []

        for row in data:
            cleaned_row = {}
            for key, value in row.items():
                if isinstance(value, str):
                    cleaned_value = value

                    if normalize_whitespace:
                        # Normalize various types of whitespace
                        cleaned_value = re.sub(r'[\r\n\t]+', ' ', cleaned_value)
                        cleaned_value = re.sub(r'\s+', ' ', cleaned_value)  # Multiple spaces to single
                        cleaned_value = cleaned_value.strip()
                    else:
                        cleaned_value = cleaned_value.strip()

                    # Truncate to max length
                    if max_length > 0:
                        cleaned_value = cleaned_value[:max_length]

                    cleaned_row[key] = cleaned_value
                else:
                    # Convert non-string values to string for consistency
                    str_value = str(value) if value is not None else ""
                    if normalize_whitespace:
                        str_value = re.sub(r'[\r\n\t]+', ' ', str_value)
                        str_value = re.sub(r'\s+', ' ', str_value)
                        str_value = str_value.strip()

                    if max_length > 0:
                        str_value = str_value[:max_length]

                    cleaned_row[key] = str_value

            cleaned_data.append(cleaned_row)

        text_columns = len(set().union(*[row.keys() for row in data])) if data else 0
        logging.debug(f"Cleaned {text_columns} columns")
        return cleaned_data

    @staticmethod
    def handle_missing_values(data: list[dict[str, Any]],
                              fill_object_values: str = "",
                              fill_numeric_values: Optional[int | float | str] = None) -> list[dict[str, Any]]:
        """
        Enhanced missing value handling with separate strategies for different types.

        Args:
            data (list[dict[str, Any]]): Input list of dictionaries.
            fill_object_values (str): Value to fill missing object/text values.
            fill_numeric_values (Union[int, float, str]): Value to fill missing numeric values (None keeps as None).

        Returns:
            list[dict[str, Any]]: List of dictionaries with missing values handled (copy).
        """
        if not data:
            return data.copy()

        cleaned_data = []

        for row in data:
            cleaned_row = {}
            for key, value in row.items():
                if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
                    # Handle missing/empty values
                    if fill_numeric_values is not None and WorksheetUtils._is_numeric_string(str(value)):
                        cleaned_row[key] = fill_numeric_values
                    else:
                        cleaned_row[key] = fill_object_values
                else:
                    cleaned_row[key] = value

            cleaned_data.append(cleaned_row)

        return cleaned_data

    @staticmethod
    def _is_numeric_string(value: str) -> bool:
        """Helper method to check if a string represents a numeric value."""
        if not value or value.strip() == "":
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def remove_unnamed_and_null_columns(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Remove columns whose names start with 'Unnamed' (common after CSV export)
        and columns where all values are null/empty.

        Args:
            data (list[dict[str, Any]]): Input list of dictionaries.

        Returns:
            list[dict[str, Any]]: List of dictionaries without unnamed or all-null columns (copy).
        """
        if not data:
            return data.copy()

        # Get all column names
        all_columns = set()
        for row in data:
            all_columns.update(row.keys())

        # Find unnamed columns
        unnamed_cols = {col for col in all_columns if str(col).startswith('Unnamed')}

        # Find columns where all values are null/empty
        null_cols = set()
        for col in all_columns:
            all_null = True
            for row in data:
                value = row.get(col)
                if value is not None and value != "" and (not isinstance(value, str) or value.strip() != ""):
                    all_null = False
                    break
            if all_null:
                null_cols.add(col)

        to_remove = unnamed_cols | null_cols
        if to_remove:
            logging.debug(f"Removing columns: {list(to_remove)}")

        # Create new data without the columns to remove
        cleaned_data = []
        for row in data:
            cleaned_row = {key: value for key, value in row.items() if key not in to_remove}
            cleaned_data.append(cleaned_row)

        return cleaned_data

    @staticmethod
    def transform_column_names(data: list[dict[str, Any]],
                               naming_convention: str = "snake_case",
                               remove_prefixes: bool = True) -> list[dict[str, Any]]:
        """
        Enhanced column name transformation with better error handling.

        Args:
            data (list[dict[str, Any]]): Input list of dictionaries.
            naming_convention (str): "snake_case" or "camelCase".
            remove_prefixes (bool): Whether to remove dot-separated prefixes.

        Returns:
            list[dict[str, Any]]: List of dictionaries with transformed column names (copy).
        """
        if not data:
            return data.copy()

        if naming_convention.lower() not in ["snake_case", "camelcase"]:
            logging.warning(f"Invalid naming_convention '{naming_convention}'. Using 'snake_case'")
            naming_convention = "snake_case"

        try:
            # Get all unique column names from the data
            all_columns = set()
            for row in data:
                all_columns.update(row.keys())

            # Create mapping from old to new column names
            column_mapping = {}
            for col in all_columns:
                col_str = str(col)

                if remove_prefixes and "." in col_str:
                    # Remove prefix (everything before last dot)
                    col_clean = col_str.split(".")[-1]
                else:
                    col_clean = col_str.replace(".", "_")

                col_clean = ''.join(
                    c for c in normalize('NFKD', col_clean)
                    if not combining(c)
                )

                col_clean = re.sub(r'[^a-zA-Z0-9_\-.\s]', '', col_clean)

                if naming_convention.lower() == "snake_case":
                    # Convert to snake_case
                    new_col = (col_clean.replace("-", "_")
                               .replace(" ", "_")
                               .lower())
                    # Clean up multiple underscores
                    new_col = re.sub(r'_+', '_', new_col).strip('_')

                elif naming_convention.lower() == "camelcase":
                    # Convert to camelCase
                    parts = re.split(r'[.\-_\s]+', col_clean)
                    new_col = parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])

                column_mapping[col] = new_col

            # Apply the column name transformation
            transformed_data = []
            for row in data:
                transformed_row = {column_mapping.get(key, key): value for key, value in row.items()}
                transformed_data.append(transformed_row)

            logging.debug(f"Transformed column names to {naming_convention}")
            return transformed_data

        except Exception as e:
            logging.warning(f"Column naming transformation failed: {e}")
            return data.copy()

    def get_data_summary(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Get a comprehensive summary of data types and quality.

        Args:
            data (list[dict[str, Any]]): Input list of dictionaries.

        Returns:
            dict[str, Any]: Dictionary with data quality metrics.
        """
        if not data:
            return {
                'total_rows': 0,
                'total_columns': 0,
                'missing_values': {},
                'numeric_columns': 0,
                'date_columns': 0,
                'text_columns': 0,
            }

        # Get all column names
        all_columns = set()
        for row in data:
            all_columns.update(row.keys())

        # Analyze data types and missing values
        missing_values = {col: 0 for col in all_columns}
        numeric_cols = set()
        date_cols = set()
        text_cols = set()

        for row in data:
            for col in all_columns:
                value = row.get(col)

                # Count missing values
                if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
                    missing_values[col] += 1

                # Analyze types (simple heuristics)
                if value is not None:
                    # Check if it's numeric
                    if isinstance(value, (int, float)) or self._is_numeric_string(str(value)):
                        numeric_cols.add(col)
                    elif isinstance(value, str):
                        # Simple date detection (you could make this more sophisticated)
                        if self._looks_like_date(value):
                            date_cols.add(col)
                        else:
                            text_cols.add(col)

        summary = {
            'total_rows': len(data),
            'total_columns': len(all_columns),
            'missing_values': missing_values,
            'numeric_columns': len(numeric_cols),
            'date_columns': len(date_cols),
            'text_columns': len(text_cols),
        }

        return summary

    def _looks_like_date(self, value: str) -> bool:
        """Simple heuristic to detect if a string looks like a date."""
        if not isinstance(value, str) or len(value.strip()) == 0:
            return False

        # Simple patterns for date detection
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]

        for pattern in date_patterns:
            if re.match(pattern, value.strip()):
                return True

        return False
