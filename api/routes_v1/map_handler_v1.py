from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import os

router = APIRouter()

@router.get("/route-map/{route_type}/{location_id:path}")
async def get_route(route_type: str, location_id: str):
    """
    Returns the corresponding HTML map based on the route_type and location_id.
    """
    # Call generate_route_map with the location_id.
    from services.route_optimizer_new import generate_route_map
    await generate_route_map(location_id)

    # Replace any "/" in location_id with "_" to create a safe file name.
    safe_location_id = location_id.replace("/", "_")

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
        file_name = f"maps/{route_type}_map{safe_location_id}.html"
        if not os.path.exists(file_name):
            raise HTTPException(status_code=404, detail="Route map not found.")
        try:
            with open(file_name, "r") as file:
                html_content = file.read()
            return HTMLResponse(content=html_content, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
