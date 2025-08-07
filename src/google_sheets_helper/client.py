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
        load_sheet_as_dataframe: Reads a worksheet and returns a cleaned DataFrame.
        _get_drive_file_mime_type: Returns the mimeType of a file in Google Drive using the service account.

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

            credentials = Credentials.from_service_account_info(client_secret, scopes=scopes)
            self.gc = gspread.authorize(credentials)
            self.service = build('drive', 'v3', credentials=credentials, cache_discovery=False)

            logging.info("Google Sheets service account authentication successful.")

        except Exception as e:
            logging.error(f"Google Sheets authentication failed: {e}", exc_info=True)
            raise AuthenticationError("Failed to authenticate with Google Sheets API", original_error=e) from e

    def load_sheet_as_dataframe(self, file_id: str, worksheet_name: str = None,
                                header_row: int = 1, log_columns: bool = True) -> pd.DataFrame:
        """
        Loads a Google Sheet or Excel file from Google Drive and returns a cleaned DataFrame.

        Parameters:
            file_id (str): The file ID in Google Drive (for both Google Sheets and Excel).
            worksheet_name (str): The name of the worksheet/tab to read.
            header_row (int): The row number (1-based) containing column headers.
            log_columns (bool): Whether to add log columns for tracking.

        Returns:
            pd.DataFrame: Cleaned DataFrame with parsed and transformed data, plus log columns.

        Raises:
            DataProcessingError: If reading or parsing the sheet fails.
        """
        try:
            mime_type = self._get_drive_file_mime_type(file_id)

            # Google Sheets
            if mime_type == "application/vnd.google-apps.spreadsheet":
                sh: gspread.Spreadsheet = self.gc.open_by_key(file_id=file_id)
                spreadsheet_title = sh.title

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
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]:
                suffix = '.xls' if mime_type == "application/vnd.ms-excel" else '.xlsx'

                file_metadata = self.service.files().get(fileId=file_id, fields="name").execute()
                spreadsheet_title = file_metadata.get("name", "")

                request = self.service.files().get_media(fileId=file_id)

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

            if log_columns and not df.empty:
                df['spreadsheet_key'] = file_id
                df['file_name'] = f"{spreadsheet_title}_{worksheet_name}"
                df['read_at'] = pd.Timestamp.now()

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
            file = self.service.files().get(fileId=file_id, fields="mimeType").execute()
            return file.get("mimeType", "")

        except Exception as e:
            logging.error(f"Failed to get mimeType for file {file_id}: {e}", exc_info=True)
            raise DataProcessingError(f"Failed to get mimeType for file {file_id}", original_error=e)
