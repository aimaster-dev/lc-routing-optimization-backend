from fastapi import APIRouter,Query, UploadFile, HTTPException, File
from services.csv_service import save_file, process_csv, get_details_by_route_formatted
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import io
import os
from services.download_service import compress_directory, make_data_for_download, copy_directory
import shutil

router = APIRouter()

@router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Uploads a single CSV file, processes it, and returns the extracted data.
    """
    # Validate that the file is a CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a CSV file.")

    try:
        # Save the file and process it
        file_path = save_file(file)
        extracted_data = process_csv(file_path)
        #for download function
        temp_data = set(map(lambda element : element['route_number'], extracted_data))
        download_file_data = {'Route_ID': list(temp_data)}
        os.makedirs("services/route_optimization_output/Sequence", exist_ok=True)
        os.makedirs("services/route_optimization_output/IND_results", exist_ok=True)
        pd.DataFrame(download_file_data).to_csv('Benefits_new_data.csv', index=False)
        return {
            "message": "File processed successfully.",
            "file_name": file.filename,
            "file_path": file_path,
            "data": extracted_data
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/get-route-details/{route_number:path}")
async def get_route_details_formatted(route_number: str):
    """
    Fetches all rows for a given Route Number from the hard-coded CSV file.
    Returns the data in the specified format.
    """
    file_name = "uploaded_files/transformed_data_snowflk.csv" 

    try:
        # Retrieve formatted data by Route Number
        details = get_details_by_route_formatted(file_name, route_number)
        return {
            "message": "Records found.",
            "route": details
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
@router.get("/download-csv")
async def download_csv(filename: str = Query("sequence_row.csv", description="The CSV filename to download")):
    # Build the file path using the provided filename.
    csv_file_path = f'./{filename}'
    
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")
    
    # Convert the DataFrame to CSV string ( the index)
    csv_output = df.to_csv(index=False)
    
    # Create an in-memory buffer to stream the CSV data.
    buffer = io.StringIO(csv_output)
    
    return StreamingResponse(
        buffer, 
        media_type="text/csv", 
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/download-full")
async def download_csv():
    await make_data_for_download(2)
    print("copying mapping....")
    await copy_directory("maps", "services/route_optimization_output/maps")
    await compress_directory("services/route_optimization_output", "services/result_download")
    if os.path.exists("services/route_optimization_output"):
        shutil.rmtree("services/route_optimization_output")
    return FileResponse("services/result_download.zip", media_type='application/zip', filename="download.zip")