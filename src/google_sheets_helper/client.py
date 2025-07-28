
"""
Google Sheets Helper client module.

This module contains the main GoogleSheetsHelper class for reading Google Sheets and converting to DataFrames.
"""

import logging

import gspread
import pandas as pd
import tempfile

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .exceptions import AuthenticationError, DataProcessingError


class GoogleSheetsHelper:
    """
    GoogleSheetsHelper class for reading Google Sheets and converting to DataFrames.

    This class enables reading Google Sheets using a service account, parsing the data,
    converting it to a pandas DataFrame, and applying data cleaning and transformation routines.

    Parameters:
        credentials_path (str): Path to the service account JSON credentials file.

    Methods:
        read_sheet_to_df: Reads a worksheet and returns a cleaned DataFrame.
        _fix_data_types: Optimizes column data types (dates, numerics).
        _handle_missing_values: Handles missing values by column type.
        _clean_text_encoding: Cleans text columns for encoding issues.
    """

    def __init__(self, client_secret: dict):
        """
        Initializes the GoogleSheetsHelper instance and authenticates with Google Sheets API.

        Parameters:
            credentials (str): Dict with service account credentials JSON content.

        Raises:
            AuthenticationError: If credentials are invalid or authentication fails
        """
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly"
            ]

            self.credentials = Credentials.from_service_account_info(client_secret, scopes=scopes)
            self.gc = gspread.authorize(self.credentials)

            logging.info("Google Sheets service account authentication successful.")

        except Exception as e:
            logging.error(f"Google Sheets authentication failed: {e}", exc_info=True)
            raise AuthenticationError("Failed to authenticate with Google Sheets API", original_error=e) from e

    def read_sheet_to_df(self, file_id: str, worksheet_name: str = None, header_row: int = 1) -> pd.DataFrame:
        """
        Reads a Google Sheet or Excel file from Google Drive and returns a cleaned DataFrame.

        Parameters:
            file_id (str): The file ID in Google Drive (for both Google Sheets and Excel).
            worksheet_name (str): The name of the worksheet/tab to read.
            header_row (int): The row number (1-based) containing column headers.

        Returns:
            pd.DataFrame: Cleaned DataFrame with parsed and transformed data.

        Raises:
            DataProcessingError: If reading or parsing the sheet fails.
        """
        try:
            mime_type = self._get_drive_file_mime_type(file_id)

            # Google Sheets
            if mime_type == "application/vnd.google-apps.spreadsheet":
                sh: gspread.Spreadsheet = self.gc.open_by_key(file_id=file_id)

                if worksheet_name is not None:
                    worksheet = sh.worksheet(worksheet_name)
                else:
                    worksheet = sh.get_worksheet(0)  # First worksheet

                data = worksheet.get_all_values()
                if not data or len(data) < header_row:
                    raise DataProcessingError("Sheet is empty or header row is missing.")

                headers = data[header_row - 1]
                rows = data[header_row:]
                df = pd.DataFrame(rows, columns=headers)

            # Excel (xlsx or xls)
            elif mime_type in [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel"
            ]:
                suffix = '.xls' if mime_type == "application/vnd.ms-excel" else '.xlsx'

                service = build('drive', 'v3', credentials=self.credentials)
                request = service.files().get_media(fileId=file_id)

                with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp_file:
                    downloader = MediaIoBaseDownload(tmp_file, request)
                    done = False

                    while not done:
                        _, done = downloader.next_chunk()
                    tmp_file.flush()

                    # If worksheet_name is provided, use it; else, use the first sheet
                    df = pd.read_excel(
                        tmp_file.name,
                        sheet_name=worksheet_name if worksheet_name else 0,
                        header=header_row-1
                    )

            else:
                raise DataProcessingError(f"Unsupported file type: {mime_type}")

            # Apply cleaning routines
            df = self._fix_data_types(df)
            df = self._handle_missing_values(df)
            df = self._clean_text_encoding(df)

            return df

        except Exception as e:
            logging.error(f"Failed to read or parse file from Drive: {e}", exc_info=True)
            raise DataProcessingError("Failed to read or parse file from Drive", original_error=e) from e

    def _get_drive_file_mime_type(self, file_id: str) -> str:
        """
        Returns the mimeType of a file in Google Drive using the service account.
        """
        try:
            # Reuse credentials if already present
            creds = self.credentials
            service = build('drive', 'v3', credentials=creds)
            file = service.files().get(fileId=file_id, fields="mimeType").execute()
            return file.get("mimeType", "")

        except Exception as e:
            logging.error(f"Failed to get mimeType for file {file_id}: {e}", exc_info=True)
            raise DataProcessingError(f"Failed to get mimeType for file {file_id}", original_error=e)

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempts to convert columns to appropriate data types (dates, numerics).
        """
        for col in df.columns:
            # Try to parse dates
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except Exception:
                pass
            # Try to parse numerics
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        return df

    def _handle_missing_values(self, df: pd.DataFrame, fill_object_values: str = "") -> pd.DataFrame:
        """
        Handles missing values in the DataFrame.
        """
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].replace({None: fill_object_values, "": fill_object_values})
            else:
                df[col] = df[col].replace({None: pd.NA, "": pd.NA})
        return df

    def _clean_text_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans text columns for encoding issues and trims whitespace.
        """
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(r'[\r\n]+', ' ', regex=True)
            df[col] = df[col].str.strip()
            df[col] = df[col].str[:255]
        return df
