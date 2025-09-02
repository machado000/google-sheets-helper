"""
Basic Google Sheets Helper Example

This example demonstrates the basic usage of the google-sheets-helper package
to extract data from Google Sheets and export it to a CSV file without pandas.
"""
import csv
import logging
import os

from google_sheets_helper import GoogleSheetsHelper, WorksheetUtils, load_client_secret


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():

    # Load credentials
    client_secret = load_client_secret()
    gs_helper = GoogleSheetsHelper(client_secret)

    # Spreadsheet and worksheet info
    spreadsheet_id = "1dtc_YaDRa1JO_-_CE3wNWhdrQEeW3gXs"
    worksheet_name = "Receita"

    # Load data as list of dictionaries
    # ws_data = gs_helper.load_sheet_as_dict(spreadsheet_id, worksheet_name)
    filepath = 'data/Material_-_Campanha_08_2025.xlsx'
    ws_data = gs_helper.load_excel_as_dict(filepath, worksheet_name)

    if ws_data:
        # Initialize utilities
        utils = WorksheetUtils()

        # Clean and transform data
        ws_data = utils.remove_unnamed_and_null_columns(ws_data)
        ws_data = utils.transform_column_names(ws_data, naming_convention="snake_case")
        ws_data = utils.clean_text_encoding(ws_data)

        # Get data summary
        summary = utils.get_data_summary(ws_data)
        print("\n=== Data Summary ===")
        print(f"📊 Total rows: {summary['total_rows']:,}")
        print(f"📋 Total columns: {summary['total_columns']}")
        print(f"🔢 Numeric columns: {summary['numeric_columns']}")
        print(f"📅 Date columns: {summary['date_columns']}")
        print(f"📝 Text columns: {summary['text_columns']}")

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        filename = os.path.join("data", f"{spreadsheet_id}_{worksheet_name}.csv")

        if ws_data:
            # Get all column names from the data
            all_columns = set()
            for row in ws_data:
                all_columns.update(row.keys())

            # Write to CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_columns))
                writer.writeheader()
                writer.writerows(ws_data)

            print(f"\nData saved to: {filename}")
    else:
        print("No data found or unsupported file type")


if __name__ == "__main__":
    main()
