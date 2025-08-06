"""
Basic Google Sheets Helper Example

This example demonstrates the basic usage of the google-sheets-helper package
to extract data from Google Sheets and export it to a parquet file with database-like schema.
"""
import logging
import pandas as pd  # noqa
# import parquet

from google_sheets_helper import GoogleSheetsHelper, load_client_secret, setup_logging


if __name__ == "__main__":

    setup_logging(level=logging.INFO)

    # credentials_path = os.path.join(os.path.dirname(__file__), '../secrets/client_secret.json')
    client_secret = load_client_secret()
    gs_helper = GoogleSheetsHelper(client_secret)

    # Spreadsheet and worksheet info
    spreadsheet_id = "1KurBS2TTaWsDvR9aNGkqMtSgaDyBvKW8"
    worksheet_name = "Receita"

    df = gs_helper.read_sheet_to_df(spreadsheet_id, worksheet_name)

    print(df.head())
