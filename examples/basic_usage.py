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
    data = gs_helper.load_sheet_as_json(spreadsheet_id, worksheet_name)

    if data:
        # Initialize utilities
        utils = WorksheetUtils()

        # Clean and transform the data
        data = utils.clean_text_encoding(data)
        data = utils.handle_missing_values(data)
        data = utils.transform_column_names(data, naming_convention="snake_case")

        # Get data summary
        summary = utils.get_data_summary(data)

        # Simple dictionary pretty print using basic loops
        print("\n=== Data Summary ===")
        print(f"📊 Total rows: {summary['total_rows']:,}")
        print(f"📋 Total columns: {summary['total_columns']}")
        print(f"🔢 Numeric columns: {summary['numeric_columns']}")
        print(f"📅 Date columns: {summary['date_columns']}")
        print(f"📝 Text columns: {summary['text_columns']}")

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        filename = os.path.join("data", f"{spreadsheet_id}_{worksheet_name}.csv")

        if data:
            # Get all column names from the data
            all_columns = set()
            for row in data:
                all_columns.update(row.keys())

            # Write to CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_columns))
                writer.writeheader()
                writer.writerows(data)

            print(f"\nData saved to: {filename}")
    else:
        print("No data found or unsupported file type")


if __name__ == "__main__":
    main()
