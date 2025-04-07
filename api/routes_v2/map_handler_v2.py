from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import os
from pydantic import BaseModel
from typing import List

router = APIRouter()

class Coordinates(BaseModel):
    Latitude: float
    Longitude: float

class Stop(BaseModel):
    Latitude: float
    Longitude: float
    CURRENT_CONTAINER_SIZE: int
    SERVICE_WINDOW_TIME: float
    SERVICE_TYPE_CD: str
    PERM_NOTES: str

class RouteData(BaseModel):
    Haul: Coordinates
    LandFill: Coordinates
    Stops: List[Stop]

@router.post("/route-map-old")
async def get_route(data: RouteData):
    """
    Returns the corresponding HTML map based on the route_type and location_id.
    """
    from services.route_optimizer_old import generate_route_map

    await generate_route_map(data.model_dump())
    manual_file_name = f"maps/manual_map.html"
    # optimal_file_name = f"maps/optimal_map_old.html"
    optimal_file_name = f"maps/optimal_map.html"
    try:
        with open(manual_file_name, "r") as manual_file:
            manual_html_content = manual_file.read()
        with open(optimal_file_name, "r") as optimal_file:
            optimal_html_content = optimal_file.read()
        return {
            "html_manual": manual_html_content,
            "html_optimal": optimal_html_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/route-map/{route_type}/{location_id:path}")
async def get_route(route_type: str, location_id: str):
    """
    Returns the corresponding HTML map based on the route_type and location_id.
    """
    # Call generate_route_map with the location_id.
    from services.route_optimizer_old import generate_route_map
    await generate_route_map(location_id)

    # Replace any "/" in location_id with "_" to create a safe file name.
    safe_location_id = location_id.replace("/", "-")

    if route_type == "comparison":
        manual_file_name = f"maps/manual_map{safe_location_id}.html"
        optimal_file_name = f"maps/optimal_map{safe_location_id}.html"
        try:
            with open(manual_file_name, "r") as manual_file:
                manual_html_content = manual_file.read()
            with open(optimal_file_name, "r") as optimal_file:
                optimal_html_content = optimal_file.read()
            return {
                "html_manual": manual_html_content,
                "html_optimal": optimal_html_content
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    else:
        file_name = f"maps/{route_type}_map_{safe_location_id}.html"
        if not os.path.exists(file_name):
            raise HTTPException(status_code=404, detail="Route map not found.")
        try:
            with open(file_name, "r") as file:
                html_content = file.read()
            return HTMLResponse(content=html_content, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
