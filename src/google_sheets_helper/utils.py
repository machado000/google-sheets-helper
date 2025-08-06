"""
Utility functions manipulate dataframes and setup logging.
"""
import logging
import os
import re
import pandas as pd

from typing import Any, Optional, Union

import json

from .exceptions import ConfigurationError


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


def setup_logging(level: int = logging.INFO,
                  format_string: Optional[str] = None) -> None:
    """
    Setup logging configuration (affects root logger).

    Args:
        level (int): Logging level (default: INFO).
        format_string (Optional[str]): Custom format string.

    Returns:
        None
    """
    if format_string is None:
        format_string = '%(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
        ]
    )


class DataframeUtils:
    """
    Utility class for DataFrame operations with enhanced data type detection and cleaning.

    Example usage:
        utils = DataFrameUtils()
        df = utils.fix_data_types(df)
        df = utils.clean_text_encoding(df)
        df = utils.handle_missing_values(df)
        df = utils.transform_column_names(df, naming_convention="snake_case")
    """

    def __init__(self,
                 date_detection_sample_size: int = 1000,
                 numeric_detection_sample_size: int = 1000,
                 date_success_threshold: float = 0.7,
                 numeric_success_threshold: float = 0.8):
        """
        Initialize DataFrameUtils with configurable parameters.

        Args:
            date_detection_sample_size: Number of values to sample for date detection
            numeric_detection_sample_size: Number of values to sample for numeric detection
            date_success_threshold: Minimum success rate for date pattern matching
            numeric_success_threshold: Minimum success rate for numeric detection
        """
        self.date_sample_size = date_detection_sample_size
        self.numeric_sample_size = numeric_detection_sample_size
        self.date_threshold = date_success_threshold
        self.numeric_threshold = numeric_success_threshold

    def __repr__(self):
        return (f"<DataFrameUtils(date_sample={self.date_sample_size}, "
                f"numeric_sample={self.numeric_sample_size})>")

    def fix_data_types(self, df: pd.DataFrame,
                       skip_columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Optimizes data types for database storage with dynamic detection.

        Args:
            df (pd.DataFrame): Input DataFrame.
            skip_columns (Optional[list[str]]): List of column names to skip during type conversion.

        Returns:
            pd.DataFrame: DataFrame with optimized data types (copy).
        """
        if df.empty:
            logging.warning("Empty DataFrame provided to fix_data_types")
            return df.copy()

        df = df.copy()
        skip_columns = skip_columns or []

        try:
            # 1. Dynamically identify date columns
            date_columns = self._identify_date_columns(df, skip_columns)
            self._convert_date_columns(df, date_columns)

            # 2. Convert numeric columns (excluding date columns and skipped columns)
            numeric_candidates = self._identify_numeric_columns(df, date_columns, skip_columns)
            self._convert_numeric_columns(df, numeric_candidates)

            logging.info(f"Data type conversion completed. Dates: {len(date_columns)}, "
                         f"Numerics: {len(numeric_candidates)}")
            return df

        except Exception as e:
            logging.error(f"Data type optimization failed: {e}")
            return df

    def _identify_date_columns(self, df: pd.DataFrame,
                               skip_columns: list[str]) -> list[str]:
        """
        Identify columns in a DataFrame that likely contain date/datetime data.

        Args:
            df (pd.DataFrame): Input DataFrame.
            skip_columns (list[str]): List of column names to skip.

        Returns:
            list[str]: List of column names identified as date columns.
        """
        date_columns = []

        for col in df.columns:
            if col in skip_columns:
                continue

            if df[col].dtype == 'object' and not df[col].empty:
                if self._looks_like_date_column(df[col]):
                    date_columns.append(col)

        logging.debug(f"Identified date columns: {date_columns}")
        return date_columns

    def _convert_date_columns(self, df: pd.DataFrame, date_columns: list[str]) -> None:
        """
        Convert identified date columns in a DataFrame to datetime type.

        Args:
            df (pd.DataFrame): Input DataFrame (modified in place).
            date_columns (list[str]): List of column names to convert.

        Returns:
            None
        """
        for col in date_columns:
            try:
                # Use 'mixed' format to handle various date formats
                df[col] = pd.to_datetime(df[col], errors='raise', format='mixed')
                logging.debug(f"Converted {col} to datetime")
            except Exception as e:
                logging.warning(f"Could not convert {col} to datetime: {e}")

    def _identify_numeric_columns(self, df: pd.DataFrame,
                                  date_columns: list[str],
                                  skip_columns: list[str]) -> list[str]:
        """
        Identify columns in a DataFrame that should be converted to numeric types.

        Args:
            df (pd.DataFrame): Input DataFrame.
            date_columns (list[str]): List of columns already identified as dates.
            skip_columns (list[str]): List of column names to skip.

        Returns:
            list[str]: List of column names identified as numeric candidates.
        """
        numeric_candidates = []

        for col in df.columns:
            if (col not in date_columns and
                col not in skip_columns and
                df[col].dtype == 'object' and
                    not df[col].empty):

                if self._looks_numeric(df[col]):
                    numeric_candidates.append(col)

        logging.debug(f"Identified numeric columns: {numeric_candidates}")
        return numeric_candidates

    def _convert_numeric_columns(self, df: pd.DataFrame, numeric_columns: list[str]) -> None:
        """
        Convert identified numeric columns in a DataFrame to appropriate numeric types (int64 or float64).

        Args:
            df (pd.DataFrame): Input DataFrame (modified in place).
            numeric_columns (list[str]): List of column names to convert.

        Returns:
            None
        """
        for col in numeric_columns:
            try:
                # Pre-process the strings to remove currency symbols and commas
                processed_series = df[col].astype(str).str.replace(r'[,$%]', '', regex=True).str.strip()
                numeric_series = pd.to_numeric(processed_series, errors='coerce')

                # Check if all values were successfully converted
                if numeric_series.notna().all():
                    if self._should_be_integer(numeric_series):
                        df[col] = numeric_series.astype('int64')
                        logging.debug(f"Converted {col} to int64")
                    else:
                        df[col] = numeric_series.astype('float64')
                        logging.debug(f"Converted {col} to float64")
                else:
                    # If some values failed to convert, leave the column as is
                    logging.warning(f"Could not convert all values in {col} to numeric. "
                                    f"Leaving as object.")

            except (ValueError, TypeError) as e:
                logging.warning(f"Could not convert {col} to numeric: {e}")

    def _looks_like_date_column(self, series: pd.Series) -> bool:
        """
        Analyze a Series to determine if it likely contains date/datetime data.

        Args:
            series (pd.Series): Input Series.

        Returns:
            bool: True if the series likely contains date data, False otherwise.
        """
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return False

        sample = non_null_values.head(min(self.date_sample_size, len(non_null_values)))

        # Try multiple detection strategies
        strategies = [
            self._check_date_patterns(sample),
            self._check_pandas_datetime_success(sample),
            self._check_timestamp_patterns(sample)
        ]

        return any(strategies)

    def _check_date_patterns(self, sample: pd.Series) -> bool:
        """
        Check if values in a sample Series match common date patterns using regex.

        Args:
            sample (pd.Series): Sample of values from a column.

        Returns:
            bool: True if enough values match date patterns, False otherwise.
        """
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',                      # YYYY-MM-DD
            r'^\d{4}/\d{2}/\d{2}$',                      # YYYY/MM/DD
            r'^\d{2}/\d{2}/\d{4}$',                      # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',                      # MM-DD-YYYY
            r'^\d{1,2}/\d{1,2}/\d{4}$',                  # M/D/YYYY
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',     # ISO datetime
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',     # YYYY-MM-DD HH:MM:SS
            r'^[A-Za-z]{3} \d{1,2}, \d{4}$',            # Jan 15, 2024
            r'^\d{1,2}-[A-Za-z]{3}-\d{4}$',             # 15-Jan-2024
            r'^\d{4}\d{2}\d{2}$',                        # YYYYMMDD
        ]

        if len(sample) == 0:
            return False

        matches = 0
        for value in sample:
            if isinstance(value, str):
                value_clean = value.strip()
                if any(re.match(pattern, value_clean) for pattern in date_patterns):
                    matches += 1

        return (matches / len(sample)) >= self.date_threshold

    def _check_pandas_datetime_success(self, sample: pd.Series) -> bool:
        """
        Test if pandas can successfully parse values in a sample Series as dates.

        Args:
            sample (pd.Series): Sample of values from a column.

        Returns:
            bool: True if enough values can be parsed as dates, False otherwise.
        """
        if len(sample) == 0:
            return False
        try:
            # Use 'mixed' format to handle various date formats
            converted = pd.to_datetime(sample, errors='coerce', format='mixed')
            successful_conversions = converted.notna().sum()

            success_rate = successful_conversions / len(sample)

            return bool(success_rate >= self.date_threshold)

        except Exception:
            return False

    def _check_timestamp_patterns(self, sample: pd.Series) -> bool:
        """
        Check for Unix timestamps or other numeric date representations in a sample Series.

        Args:
            sample (pd.Series): Sample of values from a column.

        Returns:
            bool: True if enough values match timestamp patterns, False otherwise.
        """
        try:
            numeric_sample = pd.to_numeric(sample, errors='coerce')
            numeric_values = numeric_sample.dropna()

            if len(numeric_values) < len(sample) * 0.5:  # Ensure most are numeric
                return False

            # Check for values that are clearly not timestamps
            if numeric_values.abs().max() < 1000:  # Unlikely to be a real timestamp
                return False

            # Unix timestamps (seconds) - expanded range
            unix_min = 946684800  # 2000-01-01
            unix_max = 2524608000  # 2050-01-01

            # Millisecond timestamps
            ms_min = unix_min * 1000
            ms_max = unix_max * 1000

            unix_matches = ((numeric_values >= unix_min) &
                            (numeric_values <= unix_max)).sum()
            ms_matches = ((numeric_values >= ms_min) &
                          (numeric_values <= ms_max)).sum()

            total_values = len(numeric_values)
            unix_rate = unix_matches / total_values
            ms_rate = ms_matches / total_values

            return bool(unix_rate >= 0.8 or ms_rate >= 0.7)

        except Exception:
            return False

    def _looks_numeric(self, series: pd.Series) -> bool:
        """
        Heuristic to check if a Series might contain numeric data.

        Args:
            series (pd.Series): Input Series.

        Returns:
            bool: True if the series likely contains numeric data, False otherwise.
        """
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return False

        sample = non_null_values.head(min(self.numeric_sample_size, len(non_null_values)))
        numeric_count = 0

        for value in sample:
            if isinstance(value, (int, float)):
                numeric_count += 1
            elif isinstance(value, str):
                value_clean = value.strip()
                if value_clean and self._is_numeric_string(value_clean):
                    numeric_count += 1

        return (numeric_count / len(sample)) >= self.numeric_threshold

    def _is_numeric_string(self, value: str) -> bool:
        """
        Check if a string represents a numeric value.

        Args:
            value (str): Input string value.

        Returns:
            bool: True if the string represents a numeric value, False otherwise.
        """
        if value is None:
            return False
        try:
            float(value)
            return True
        except ValueError:
            # Check for common numeric patterns with commas, currency symbols, etc.
            cleaned = re.sub(r'[,$%\s]', '', value)
            try:
                float(cleaned)
                return True
            except ValueError:
                return False

    @staticmethod
    def _should_be_integer(numeric_series: pd.Series) -> bool:
        """
        Determine if a numeric Series should be stored as integer or float.

        Args:
            numeric_series (pd.Series): Numeric Series to check.

        Returns:
            bool: True if all values are whole numbers and fit in int64, False otherwise.
        """
        clean_series = numeric_series.dropna()

        if len(clean_series) == 0:
            return False

        # Check if all values are whole numbers
        if not (clean_series % 1 == 0).all():
            return False

        # Check int64 range to prevent overflow
        int64_min, int64_max = -2**63, 2**63 - 1
        return bool(((clean_series >= int64_min).all() and
                     (clean_series <= int64_max).all()))

    def clean_text_encoding(self, df: pd.DataFrame,
                            max_length: int = 255,
                            normalize_whitespace: bool = True) -> pd.DataFrame:
        """
        Enhanced text cleaning with configurable options.

        Args:
            df (pd.DataFrame): Input DataFrame.
            max_length (int): Maximum length for text fields.
            normalize_whitespace (bool): Whether to normalize whitespace.

        Returns:
            pd.DataFrame: DataFrame with cleaned text columns (copy).
        """
        df = df.copy()
        text_columns = df.select_dtypes(include=['object']).columns

        for col in text_columns:
            if normalize_whitespace:
                # Normalize various types of whitespace
                df[col] = (df[col].astype(str)
                           .str.replace(r'[\r\n\t]+', ' ', regex=True)
                           .str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                           .str.strip())
            else:
                df[col] = df[col].astype(str).str.strip()

            # Truncate to max length
            if max_length > 0:
                df[col] = df[col].str[:max_length]

        logging.debug(f"Cleaned {len(text_columns)} text columns")
        return df

    def handle_missing_values(self, df: pd.DataFrame,
                              fill_object_values: str = "",
                              fill_numeric_values: Union[int, float, str] = None) -> pd.DataFrame:
        """
        Enhanced missing value handling with separate strategies for different types.

        Args:
            df (pd.DataFrame): Input DataFrame.
            fill_object_values (str): Value to fill missing object/text values.
            fill_numeric_values (Union[int, float, str]): Value to fill missing numeric values (None keeps as NA).

        Returns:
            pd.DataFrame: DataFrame with missing values handled (copy).
        """
        df = df.copy()

        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                # Handle object columns
                df[col] = df[col].fillna(fill_object_values).replace("", fill_object_values)
            elif pd.api.types.is_numeric_dtype(df[col]) and fill_numeric_values is not None:
                # Handle numeric columns if fill value specified
                df[col] = df[col].fillna(fill_numeric_values)
            # Leave other types as-is (datetime, etc.)

        return df

    def transform_column_names(self, df: pd.DataFrame,
                               naming_convention: str = "snake_case",
                               remove_prefixes: bool = True) -> pd.DataFrame:
        """
        Enhanced column name transformation with better error handling.

        Args:
            df (pd.DataFrame): Input DataFrame.
            naming_convention (str): "snake_case" or "camelCase".
            remove_prefixes (bool): Whether to remove dot-separated prefixes.

        Returns:
            pd.DataFrame: DataFrame with transformed column names (copy).
        """
        df = df.copy()

        if naming_convention.lower() not in ["snake_case", "camelcase"]:
            logging.warning(f"Invalid naming_convention '{naming_convention}'. Using 'snake_case'")
            naming_convention = "snake_case"

        try:
            new_columns = []

            for col in df.columns:
                if remove_prefixes and "." in col:
                    # Remove prefix (everything before last dot)
                    col_clean = col.split(".")[-1]
                else:
                    col_clean = col.replace(".", "_")

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

                new_columns.append(new_col)

            df.columns = new_columns
            logging.debug(f"Transformed column names to {naming_convention}")
            return df

        except Exception as e:
            logging.warning(f"Column naming transformation failed: {e}")
            return df

    def get_data_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get a comprehensive summary of DataFrame data types and quality.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            dict[str, Any]: Dictionary with data quality metrics.
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Add type-specific insights
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        object_cols = df.select_dtypes(include=['object']).columns

        summary.update({
            'numeric_columns': len(numeric_cols),
            'date_columns': len(date_cols),
            'text_columns': len(object_cols),
        })

        return summary
