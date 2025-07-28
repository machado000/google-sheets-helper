import os
from google_sheets_helper.client import GoogleSheetsHelper

# Path to your service account credentials
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), '../secrets/client_secret.json')


if __name__ == "__main__":

    # Spreadsheet and worksheet info
    spreadsheet_id = "1KurBS2TTaWsDvR9aNGkqMtSgaDyBvKW8"
    worksheet_name = "Receita"

    helper = GoogleSheetsHelper(credentials_path=CREDENTIALS_PATH)

    df = helper.read_sheet_to_df(spreadsheet_id, worksheet_name)

    print(df.head())
