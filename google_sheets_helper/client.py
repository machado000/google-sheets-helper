"""
Google Sheets Helper client module.

This module contains the main GoogleSheetsHelper class for reading Google Sheets and converting to DataFrames.
"""

import logging

import gspread
import pandas as pd
import tempfile
import time

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm
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

    def load_sheet_as_dataframe(self, file_id: str, worksheet_name: str,
                                header_row: int = 1, decimal: str = '.',
                                log_columns: bool = True) -> pd.DataFrame | None:
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

            valid_mime_types = [
                "application/vnd.google-apps.spreadsheet",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ]

            file_title, mime_type, _ = self._get_drive_file_metadata(file_id)

            if mime_type not in valid_mime_types:
                logging.info(f"Unsupported file type: {mime_type}")
                return None

            worksheet = None
            df = pd.DataFrame()

            # Google Sheets
            if mime_type == "application/vnd.google-apps.spreadsheet":
                sh: gspread.Spreadsheet = self.gc.open_by_key(key=file_id)

                if worksheet_name:
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
                request = self.service.files().get_media(fileId=file_id)

                with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp_file:
                    downloader = MediaIoBaseDownload(tmp_file, request, chunksize=256*1024)  # 256KB chunks
                    done = False
                    pbar = tqdm(total=100, desc="Downloading file")

                    MAX_RETRIES, RETRY_DELAY = 3, 2

                    while not done:
                        for attempt in range(MAX_RETRIES):
                            try:
                                status, done = downloader.next_chunk()
                                break  # Success, exit retry loop
                            except Exception as e:
                                if attempt < MAX_RETRIES - 1:
                                    logging.warning(
                                        f"Chunk download failed (attempt {attempt+1}/{MAX_RETRIES}), retrying: {e}")
                                    time.sleep(RETRY_DELAY)
                                else:
                                    logging.error("Max retries reached for chunk download.")
                                    raise  # Re-raise the last exception
                        if status:
                            pbar.n = int(status.progress() * 100)
                            pbar.refresh()

                    pbar.close()
                    tmp_file.flush()

                    df = pd.read_excel(
                        tmp_file.name,
                        sheet_name=worksheet_name if worksheet_name else 0,
                        header=header_row-1,
                        decimal=decimal
                    )

            else:
                raise DataProcessingError(f"Unsupported file type: {mime_type}")

            if log_columns and not df.empty:
                df['spreadsheet_key'] = file_id
                df['file_name'] = f"{file_title}_{worksheet_name}"
                df['read_at'] = pd.Timestamp.now()

            return df

        except Exception as e:
            logging.error(f"Failed to read or parse file from Drive: {e}", exc_info=True)
            raise DataProcessingError("Failed to read or parse file from Drive", original_error=e) from e

    def _get_drive_file_metadata(self, file_id: str) -> tuple:
        """
        Returns the (name, mimeType) of a file in Google Drive using the service account.

        Parameters:
            file_id (str): The ID of the file in Google Drive.

        Returns:
            tuple: (file_title, mime_type)

        Raises:
            DataProcessingError: If metadata retrieval fails.
        """
        try:
            file_metadata = self.service.files().get(fileId=file_id, fields="id,name,mimeType,size").execute()
            title = file_metadata.get("name", "")
            mime_type = file_metadata.get("mimeType", "")
            size = file_metadata.get("size", "")

            return (title, mime_type, size)

        except Exception as e:
            logging.error(f"Failed to get metadata for file {file_id}: {e}", exc_info=True)
            raise DataProcessingError(f"Failed to get metadata for file {file_id}", original_error=e)

    def list_files_in_folder(self, folder_id: str):
        """
        Lists files in a Google Drive folder.

        Parameters:
            folder_id (str): The ID of the Google Drive folder.

        Returns:
            List[Tuple[str, str]]: A list of tuples (file_id, file_name).
        """
        try:
            query = f"'{folder_id}' in parents and trashed = false"
            files = []
            page_token = None

            while True:
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name)',
                    pageToken=page_token
                ).execute()

                files.extend(response.get('files', []))

                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break

            return files

        except Exception as e:
            logging.error(f"Failed to list files in folder {folder_id}: {e}", exc_info=True)
            raise DataProcessingError(f"Failed to list files in folder {folder_id}", original_error=e)
