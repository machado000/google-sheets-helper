"""
Google Sheets Helper client module.

This module contains the main GoogleSheetsHelper class for reading Google Sheets and converting to list of dictionaries.
"""

import logging
import gspread
import tempfile

from datetime import datetime
from python_calamine import CalamineWorkbook
from typing import Any
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .exceptions import AuthenticationError, DataProcessingError


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoogleSheetsHelper:
    """
    GoogleSheetsHelper class for reading Google Sheets and converting to dictionaries.

    This class enables reading Google Sheets using a service account, parsing the data,
    converting it to a list of dictionaries, and applying data cleaning and transformation routines.

    Parameters:
        client_secret (dict): Dict with service account credentials JSON content.

    Methods:
        load_sheet_as_json: Reads a worksheet and returns a list of dictionaries.
        _get_drive_file_metadata: Returns the metadata of a file in Google Drive using the service account.

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

    def load_sheet_as_json(self, file_id: str, worksheet_name: str,
                           header_row: int = 1, log_columns: bool = True) -> list[dict[str, Any]] | None:
        """
        Loads a Google Sheet or Excel file from Google Drive and returns a list of dictionaries.

        Parameters:
            file_id (str): The file ID in Google Drive (for both Google Sheets and Excel).
            worksheet_name (str): The name of the worksheet/tab to read.
            header_row (int): The row number (1-based) containing column headers.
            log_columns (bool): Whether to add log columns for tracking.

        Returns:
            list[dict[str, Any]]: List of dictionaries with parsed and transformed data, plus log columns.

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

                # Convert to list of dictionaries with string keys
                result_data = []
                for row in rows:
                    row_dict = {str(header): str(value) for header, value in zip(headers, row)}
                    result_data.append(row_dict)

            # Excel (xlsx or xls)
            elif mime_type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]:
                suffix = '.xls' if mime_type == "application/vnd.ms-excel" else '.xlsx'
                request = self.service.files().get_media(fileId=file_id)

                with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp_file:
                    # Use larger chunks for better performance (4MB instead of 256KB)
                    downloader = MediaIoBaseDownload(tmp_file, request, chunksize=4*1024*1024)
                    done = False
                    last_logged_percent = -1

                    logging.info(f"Starting download of '{file_title}' ({mime_type})")

                    while not done:
                        try:
                            status, done = downloader.next_chunk()
                            if status:
                                current_percent = int(status.progress() * 100)
                                # Log every 10% to show progress without spam
                                if current_percent >= last_logged_percent + 10:
                                    logging.info(f"Download progress: {current_percent}%")
                                    last_logged_percent = current_percent
                        except Exception as e:
                            logging.error(f"Download failed: {e}")
                            raise DataProcessingError(f"Failed to download file: {e}")

                    logging.info("File download completed")

                    # Read Excel file using python-calamine
                    workbook = CalamineWorkbook.from_path(tmp_file.name)

                    # Get worksheet by name or index
                    if worksheet_name:
                        worksheet = workbook.get_sheet_by_name(worksheet_name)
                    else:
                        sheet_names = workbook.sheet_names
                        if not sheet_names:
                            raise DataProcessingError("No worksheets found in Excel file.")
                        worksheet = workbook.get_sheet_by_index(0)

                    if worksheet is None:
                        raise DataProcessingError(f"Worksheet '{worksheet_name}' not found in Excel file.")

                    # Read all rows from the worksheet
                    rows = list(worksheet.iter_rows())

                    if not rows or len(rows) < header_row:
                        raise DataProcessingError("Excel file is empty or header row is missing.")

                    # Get headers and data rows
                    headers = [str(cell) if cell is not None else "" for cell in rows[header_row - 1]]
                    data_rows = rows[header_row:]

                    # Convert to list of dictionaries with string keys
                    result_data = []
                    for row in data_rows:
                        row_values = [cell if cell is not None else "" for cell in row]
                        row_dict = {str(header): value for header, value in zip(headers, row_values)}
                        result_data.append(row_dict)

            else:
                raise DataProcessingError(f"Unsupported file type: {mime_type}")

            # Add log columns if requested
            if log_columns and result_data:
                for row in result_data:
                    row['spreadsheet_key'] = file_id
                    row['file_name'] = f"{file_title}_{worksheet_name}"
                    row['read_at'] = datetime.now().isoformat()

            return result_data

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
