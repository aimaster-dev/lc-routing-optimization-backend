from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from services.route_service import get_route_comparison  # This should be the updated function

router = APIRouter()

@router.get("/route-comparison/{route_number:path}")
async def get_comparison(route_number: str):
    """
    Fetches all rows for a given Route Number from the results CSV files.
    Returns a JSON object with both aggregate and sequence details.
    """
    try:
        # Retrieve formatted data by Route Number
        details = get_route_comparison(route_number)
        return JSONResponse(content={
            "message": "Records found.",
            "route": details
        })
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
