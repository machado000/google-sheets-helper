"""
Utility functions for the Google Ads driver module.
"""
import logging
import os
import re
import pandas as pd

from typing import Any, Dict, Optional

import json

from .exceptions import ConfigurationError


def load_client_secret(client_secret_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load Google Ads API credentials from JSON file.

    Args:
        config_path (Optional[str]): Path to the credentials file.
                                   If None, tries default locations.

    Returns:
        Dict[str, Any]: Loaded client_secret.json credentials

    Raises:
        FileNotFoundError: If credentials file is not found
        json.JSONDecodeError: If JSON parsing fails
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
    Setup logging configuration.

    Args:
        level (int): Logging level (default: INFO)
        format_string (Optional[str]): Custom format string
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
    Utility class for DataFrame operations.

    Example usage:
        utils = DataFrameUtils()
        df = utils.fix_data_types(df)
        df = utils.clean_text_encoding(df)
        df = utils.handle_missing_values(df)
        df = utils.transform_column_names(df, naming_convention="snake_case")
    """

    def __init__(self):
        """Initialize DataFrameUtils."""
        pass

    def __repr__(self):
        return "<DataFrameUtils>"

    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimizes data types for database storage with dynamic date detection.

        Args:
            df (pd.DataFrame): Input DataFrame
        Returns:
            pd.DataFrame: DataFrame with optimized data types
        """
        df = df.copy()
        try:
            # 1. Dynamically identify date columns
            date_columns = self._identify_date_columns(df)
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='raise')
                    logging.debug(f"Converted {col} to datetime")
                except Exception as e:
                    logging.warning(f"Could not convert {col} to datetime: {e}")
            # 2. Convert numeric columns (excluding identified date columns)
            numeric_candidates = []
            for col in df.columns:
                if col not in date_columns and df[col].dtype == 'object':
                    if self._looks_numeric(df[col]):
                        numeric_candidates.append(col)
            for col in numeric_candidates:
                try:
                    numeric_series = pd.to_numeric(df[col], errors='raise')
                    if self._should_be_integer(numeric_series):
                        df[col] = numeric_series.astype('int64')
                        logging.debug(f"Converted {col} from object to int64")
                    else:
                        df[col] = numeric_series.astype('float64')
                        logging.debug(f"Converted {col} from object to float64")
                except (ValueError, TypeError) as e:
                    logging.warning(f"Could not convert {col} to numeric: {e}")
            return df
        except Exception as e:
            logging.error(f"Data type optimization failed: {e}")
            return df

    def _identify_date_columns(self, df: pd.DataFrame, sample_size: int = 100) -> list:
        """
        Dynamically identifies columns that likely contain date/datetime data.

        Parameters:
        - df: DataFrame to analyze
        - sample_size: Number of non-null values to sample per column

        Returns:
        - list: Column names that appear to contain date data
        """
        date_columns = []

        for col in df.columns:
            if df[col].dtype == 'object':
                if self._looks_like_date_column(df[col], sample_size):
                    date_columns.append(col)

        return date_columns

    def _looks_like_date_column(self, series: pd.Series, sample_size: int = 1000) -> bool:
        """
        Analyzes a series to determine if it likely contains date data.

        Parameters:
        - series: The pandas Series to analyze
        - sample_size: Number of values to sample for analysis

        Returns:
        - bool: True if series appears to contain date data
        """
        # Get sample of non-null values
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return False

        sample = non_null_values.head(min(sample_size, len(non_null_values)))

        # Strategy 1: Pattern matching approach
        if self._check_date_patterns(sample):
            return True

        # Strategy 2: Pandas datetime parsing success rate
        if self._check_pandas_datetime_success(sample):
            return True

        # Strategy 3: Statistical analysis of potential timestamps
        if self._check_timestamp_patterns(sample):
            return True

        return False

    def _check_date_patterns(self, sample: pd.Series) -> bool:
        """
        Check if values match common date patterns using regex.
        """
        # Common date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',                    # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',                    # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',                    # MM-DD-YYYY
            r'^\d{4}/\d{2}/\d{2}$',                    # YYYY/MM/DD
            r'^\d{1,2}/\d{1,2}/\d{4}$',                # M/D/YYYY
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',   # ISO datetime
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',   # YYYY-MM-DD HH:MM:SS
            r'^[A-Za-z]{3} \d{1,2}, \d{4}$',          # Mon DD, YYYY
            r'^\d{1,2}-[A-Za-z]{3}-\d{4}$',           # DD-Mon-YYYY
        ]

        matches = 0
        for value in sample:
            if isinstance(value, str):
                value_clean = value.strip()
                for pattern in date_patterns:
                    if re.match(pattern, value_clean):
                        matches += 1
                        break

        # If 70% or more match date patterns, consider it a date column
        return (matches / len(sample)) >= 0.7

    def _check_pandas_datetime_success(self, sample: pd.Series) -> bool:
        """
        Test if pandas can successfully parse values as dates.
        """
        try:
            # Try to convert sample to datetime
            converted = pd.to_datetime(sample, errors='coerce')

            # Count successful conversions
            successful_conversions = converted.notna().sum()
            success_rate = successful_conversions / len(sample)

            # Additional check: ensure converted dates are reasonable
            if success_rate >= 0.7:
                valid_dates = converted.dropna()
                if len(valid_dates) > 0:
                    # Check if dates are within reasonable range (1900-2100)
                    min_date = pd.Timestamp('1900-01-01')
                    max_date = pd.Timestamp('2100-12-31')

                    reasonable_dates = ((valid_dates >= min_date) &
                                        (valid_dates <= max_date)).sum()
                    reasonable_rate = reasonable_dates / len(valid_dates)

                    return reasonable_rate >= 0.7

            return False

        except Exception:
            return False

    def _check_timestamp_patterns(self, sample: pd.Series) -> bool:
        """
        Check for Unix timestamps or other numeric date representations.
        """
        try:
            # Convert to numeric and check if they could be timestamps
            numeric_sample = pd.to_numeric(sample, errors='coerce')
            numeric_values = numeric_sample.dropna()

            if len(numeric_values) == 0:
                return False

            # Check for Unix timestamps (seconds since epoch)
            # Reasonable range: 1970-01-01 to 2038-01-19 (32-bit timestamp limit)
            unix_min = 0  # 1970-01-01
            unix_max = 2147483647  # 2038-01-19

            unix_timestamps = ((numeric_values >= unix_min) &
                               (numeric_values <= unix_max)).sum()

            # Check for millisecond timestamps
            ms_min = unix_min * 1000
            ms_max = unix_max * 1000

            ms_timestamps = ((numeric_values >= ms_min) &
                             (numeric_values <= ms_max)).sum()

            total_values = len(numeric_values)
            unix_match_rate = unix_timestamps / total_values
            ms_match_rate = ms_timestamps / total_values

            return unix_match_rate >= 0.8 or ms_match_rate >= 0.7

        except Exception:
            return False

    def _looks_numeric(self, series: pd.Series, sample_size: int = 1000) -> bool:
        """
        Quick heuristic to check if a series might contain numeric data.

        Parameters:
        - series: The pandas Series to check

        Returns:
        - bool: True if series looks like it might contain numeric data
        """
        # Get sample of non-null values
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return False

        # Get a sample of non-null values for testing
        sample = series.dropna().head(min(sample_size, len(non_null_values)))

        # Count how many values look numeric
        numeric_count = 0

        for value in sample:
            if isinstance(value, (int, float)):
                numeric_count += 1
            elif isinstance(value, str):
                # Check if string represents a number
                value_clean = value.strip()
                if value_clean == '':
                    continue
                try:
                    float(value_clean)
                    numeric_count += 1
                except ValueError:
                    pass

        # If at least 80% of sampled values look numeric, consider it numeric
        return (numeric_count / len(sample)) >= 0.8

    @staticmethod
    def _should_be_integer(numeric_series: pd.Series) -> bool:
        """
        Determines if a numeric series should be stored as integer or float.

        Parameters:
        - numeric_series: The pandas Series with numeric data

        Returns:
        - bool: True if should be integer, False if should be float
        """
        # Remove NaN values for analysis
        clean_series = numeric_series.dropna()

        if len(clean_series) == 0:
            return False  # Default to float if no data

        # If all values are whole numbers, check if they fit in int64 range
        if (clean_series % 1 == 0).all():
            # Check for int64 overflow to prevent the -9223372036854775808 issue
            int64_min, int64_max = -2**63, 2**63 - 1
            if (clean_series >= int64_min).all() and (clean_series <= int64_max).all():
                return True

        return False  # Has decimals or would overflow, should be float

    @staticmethod
    def clean_text_encoding(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans text columns for encoding issues and trims whitespace.
        """
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(r'[\r\n]+', ' ', regex=True)
            df[col] = df[col].str.strip()
            df[col] = df[col].str[:255]
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, fill_object_values: str = "") -> pd.DataFrame:
        """
        Handles missing values in the DataFrame.
        """
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].replace({None: fill_object_values, "": fill_object_values})
            else:
                df[col] = df[col].replace({None: pd.NA, "": pd.NA})
        return df

    @staticmethod
    def transform_column_names(df: pd.DataFrame, naming_convention: str = "snake_case") -> pd.DataFrame:
        """
        Transforms column names according to the specified naming convention.

        Parameters:
            df (pd.DataFrame): DataFrame with original column names
            naming_convention (str):
                - "snake_case": campaign.name → campaign_name (default)
                - "camelCase": campaign.name → campaignName
        Returns:
            pd.DataFrame: DataFrame with transformed column names
        """
        # Validate column naming parameter
        if naming_convention.lower() not in ["snake_case", "camelcase"]:
            naming_convention = "snake_case"
            logging.warning(f"Invalid column_naming '{naming_convention}'. Using 'snake_case' as default")

        try:
            if naming_convention.lower() == "snake_case":
                # Remove prefixes and convert to snake_case
                df.columns = [
                    col.replace(".", "_")
                       .lower()
                    for col in df.columns
                ]

            elif naming_convention.lower() == "camelcase":
                # Remove prefixes and convert to camelCase
                renamed_columns = []
                for col in df.columns:
                    # Convert to camelCase by capitalizing first letter after each dot, then removing dots
                    parts = col.split(".")
                    # Keep first part as is, capitalize first letter of subsequent parts
                    camel_case_col = parts[0] + "".join(part.capitalize() for part in parts[1:])
                    renamed_columns.append(camel_case_col)
                df.columns = renamed_columns

            return df

        except Exception as e:
            logging.warning(f"Column naming transformation failed: {e}")
            return df
