"""
Basic Google Sheets Helper Example

This example demonstrates the basic usage of the google-sheets-helper package
to extract data from Google Sheets and export it to a parquet file with database-like schema.
"""
import logging
import pandas as pd  # noqa
# import parquet

from google_sheets_helper import GoogleSheetsHelper, DataframeUtils, load_client_secret, setup_logging
import os


if __name__ == "__main__":

    setup_logging(level=logging.INFO)

    # credentials_path = os.path.join(os.path.dirname(__file__), '../secrets/client_secret.json')
    client_secret = load_client_secret()
    gs_helper = GoogleSheetsHelper(client_secret)

    # Spreadsheet and worksheet info
    spreadsheet_id = "1KurBS2TTaWsDvR9aNGkqMtSgaDyBvKW8"
    worksheet_name = "Receita"

    df = gs_helper.load_sheet_as_dataframe(spreadsheet_id, worksheet_name)

    utils = DataframeUtils()

    df = utils.fix_data_types(df, skip_columns=None)
    df = utils.handle_missing_values(df)
    df = utils.clean_text_encoding(df)
    df = utils.transform_column_names(df, naming_convention="snake_case")

    print(df.head(), df.dtypes)

    os.makedirs("data", exist_ok=True)
    filename = os.path.join("data", f"{spreadsheet_id}_{worksheet_name}.csv")

    df.to_csv(filename, index=False)
