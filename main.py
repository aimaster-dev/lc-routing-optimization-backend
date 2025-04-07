from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your routers
from api.routes_v1 import csv_handler_v1
from api.routes_v1 import map_handler_v1
from api.routes_v1 import route_handler_v1

from api.routes_v2 import csv_handler_v2
from api.routes_v2 import map_handler_v2
from api.routes_v2 import route_handler_v2

app = FastAPI()

# Set up CORS (allow all origins for local dev; adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with a prefix
app.include_router(csv_handler_v1.router, prefix="/api/v1", tags=["CSV Handler"])
app.include_router(map_handler_v1.router, prefix="/api/v1", tags=["Map Handler"])
app.include_router(route_handler_v1.router, prefix="/api/v1", tags=["Route Handler"])

app.include_router(csv_handler_v2.router, prefix="/api/v2", tags=["CSV Handler"])
app.include_router(map_handler_v2.router, prefix="/api/v2", tags=["Map Handler"])
app.include_router(route_handler_v2.router, prefix="/api/v2", tags=["Route Handler"])

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI CSV handler!"}
