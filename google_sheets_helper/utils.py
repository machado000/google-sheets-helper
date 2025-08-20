"""
Utility functions manipulate dataframes and setup logging.
"""
import json
import logging
import os
import pandas as pd
import re

from typing import Any, Optional
from unicodedata import normalize, combining
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
        df = utils.clean_text_encoding(df)
        df = utils.handle_missing_values(df)
        df = utils.transform_column_names(df, naming_convention="snake_case")
    """

    def __init__(self):
        """
        Initialize DataFrameUtils.
        """

    @staticmethod
    def clean_text_encoding(df: pd.DataFrame,
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

    @staticmethod
    def handle_missing_values(df: pd.DataFrame,
                              fill_object_values: str = "",
                              fill_numeric_values: Optional[int | float | str] = None) -> pd.DataFrame:
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
                df[col] = df[col].fillna(fill_object_values).replace("", fill_object_values)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if fill_numeric_values is not None:
                    df[col] = df[col].fillna(fill_numeric_values)
                # else: leave NaNs as-is
            # Leave other types as-is (datetime, etc.)

        return df

    @staticmethod
    def remove_unnamed_and_null_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns whose names start with 'Unnamed' (common after CSV export)
        and columns where all values are null.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame without unnamed or all-null columns (copy).
        """
        df = df.copy()
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed')]
        null_cols = [col for col in df.columns if df[col].isnull().all()]

        to_remove = set(unnamed_cols) | set(null_cols)
        if to_remove:
            logging.debug(f"Removing columns: {list(to_remove)}")

        df = df.drop(columns=list(to_remove), errors='ignore')
        return df

    @staticmethod
    def transform_column_names(df: pd.DataFrame,
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
