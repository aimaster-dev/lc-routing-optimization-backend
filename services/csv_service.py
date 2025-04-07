import os
import csv
from typing import List, Dict
from fastapi import UploadFile

UPLOAD_FOLDER = "uploaded_files"

def save_file(file: UploadFile) -> str:
    """Saves the uploaded file to the upload folder."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path

def process_csv(file_path: str) -> List[Dict[str, str]]:
    """Reads the CSV file and extracts the required columns."""
    extracted_data = []
    try:
        with open(file_path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                extracted_data.append({
                    "customer_name": row.get("Customer Name", ""),
                    "route_number": row.get("Route #", "") + '-' + row.get("DIVISION", ""),
                })
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {e}")
    return extracted_data

def get_details_by_route_formatted(file_name: str, route_number: str) -> List[Dict[str, str]]:
    """
    Reads the CSV file and retrieves all rows matching the given Route Number.
    Formats each row into the required structure.

    :param file_name: Name of the CSV file (e.g., 'new_data_2025.csv').
    :param route_number: The Route Number to filter by.
    :return: A list of dictionaries with the formatted data.
    """
    extracted_data = []
    try:
        with open(file_name, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # print(row.get("Route #"))
                if row.get("Route #") == route_number:
                    # Map the CSV row to the required format
                    extracted_data.append({
                        "ops_unit_code": "",
                        "route_type": "RO",
                        "customer_name": row.get("ACCOUNT_NAME", ""),  # Adjusted to match CSV
                        "address": row.get("ADDRESS", ""),
                        "city": row.get("CITY", ""),
                        "state": row.get("STATE", ""),
                        "state": row.get("State", ""),
                        "route_number": row.get("Route #", ""),
                        "service_date": row.get("SERVICE_DATE", ""),
                        "ticket_number": row.get("PK_ACTIVE_ROUTE_DETAILS_ID", ""),  # No "Ticket #" in CSV
                        "current_container_type": row.get("CURRENT_CONTAINER_TYPE", ""),
                        "current_container_size": row.get("CURRENT_CONTAINER_SIZE", ""),
                        "service_type_cd": row.get("SERVICE_TYPE_CD", ""),
                        "material_type_cd": row.get("DISPOSAL_CD", ""),  # No "Material Type CD" in CSV
                        "zip": row.get("ZIPCODE", ""),  # Adjusted
                        "latitude": float(row.get("Latitude", 0) or 0),  # Ensure conversion
                        "longitude": float(row.get("Longitude", 0) or 0),
                        "seq_number": int(int(float(row.get("SEQUENCE", 0))) or 0),  # Adjusted
                        "service_time": row.get("ROUTE_STOP_TIME", ""),  # No "Service Time", mapped to stop time
                        "start_time": row.get("ROUTE_START_TIME", ""),
                        "stop_time": row.get("ROUTE_STOP_TIME", ""),
                        "future_container_type": row.get("CONTAINER_GROUP", ""),  # No "Future Container Type"
                        "future_container_size": "",  # Not available in the CSV
                        "facility_fac5_unit_cd_1": row.get("DISPOSAL_CD1", ""),
                        "facility_fac5_unit_cd_2": row.get("DISPOSAL_PRICE_CD1", ""),
                        "facility_fac5_unit_cd_3": row.get("DISPOSAL_PRICE_CD", ""),
                        "charges": "",  # Not available in the CSV
                        "internal_cost": "",  # Not available in the CSV
                        "driver_code": "",  # Not available in the CSV
                        "vehicle_type": "",  # Not available in the CSV
                        "tandem_capable": "",  # Not available in the CSV
                        "night_route_allowed": "",  # Not available in the CSV
                        "container_id": "",  # Not available in the CSV
                        "gate_access_required": "",  # Not available in the CSV
                        "notes_1": row.get("NOTES1", ""),
                        "notes_2": row.get("NOTES2", ""),
                        "notes_3": row.get("NOTES3", ""),
                    })
        if not extracted_data:
            raise ValueError(f"No records found for Route Number '{route_number}'.")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    return extracted_data
