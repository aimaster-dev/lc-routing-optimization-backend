import pandas as pd
import math
import numpy as np
import googlemaps
import datetime
import random
from gurobipy import Model, GRB, quicksum
import os
import folium
from folium import plugins
from dataclasses import dataclass
from typing import List, Tuple, Dict
import requests
import copy
import re
import asyncio

# Ensure the "maps" folder exists
os.makedirs("maps", exist_ok=True)

# ===============================
# Data Classes for Stops
# ===============================
@dataclass
class Stop:
    id: str
    latitude: float
    longitude: float
    container_size: str
    name: str

@dataclass
class SwingEnabledStop(Stop):
    can_swing: bool = False
    current_container: str = ""

@dataclass
class EnhancedStop(Stop):
    can_swing: bool = False
    current_container: str = ""
    operation_type: str = "DRT"
    is_compactor: bool = False
    has_space_for_swing: bool = True
    delivery_container_size: str = ""

# ===============================
# Route Visualizer
# ===============================
class RouteVisualizer:
    OPERATION_COLORS = {
        "SWG": "#28a745",      # Green
        "DRT": "#dc3545",      # Red
        "MAIN_ROUTE": "#007bff",  # Blue
        "LANDFILL": "#6c757d",    # Gray
        "HAULING": "#007bff"      # Blue
    }
    
    @staticmethod
    def create_map(
        stops: List[Stop],
        sequence: List[Stop],
        route_info: Dict,
        swing_decisions: List[bool] = None,
        landfill_locs: List[Tuple[float, float]] = None,
        hauling_loc: Tuple[float, float] = None
    ) -> folium.Map:
        if hauling_loc is None:
            hauling_loc = (41.655032, -86.0097)
        if landfill_locs is None:
            landfill_locs = [(33.4353, -112.0065)]
        
        center_lat = (hauling_loc[0] + landfill_locs[0][0]) / 2
        center_lon = (hauling_loc[1] + landfill_locs[0][1]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        folium.LayerControl().add_to(m)
        RouteVisualizer._add_facility_markers(m, hauling_loc, landfill_locs)
        RouteVisualizer._add_route_visualization(m, sequence, hauling_loc, landfill_locs, route_info, swing_decisions)
        RouteVisualizer._add_legend(m)
        return m
    
    @staticmethod
    def _add_facility_markers(m: folium.Map, hauling_loc: Tuple[float, float], landfill_locs: List[Tuple[float, float]]):
        folium.Marker(
            hauling_loc,
            popup="Hauling Facility",
            icon=folium.DivIcon(html="""\
                <div style="font-family: courier new; color: #007bff; 
                font-size: 24px; font-weight: bold; text-align: center;
                background-color: white; border-radius: 50%; width: 30px; 
                height: 30px; line-height: 30px; border: 2px solid #007bff;">
                H</div>""")
        ).add_to(m)
        
        for i, landfill_loc in enumerate(landfill_locs):
            folium.Marker(
                landfill_loc,
                popup=f"Landfill {i + 1}",
                icon=folium.DivIcon(html=f"""\ 
                    <div style="font-family: courier new; color: #6c757d; 
                    font-size: 24px; font-weight: bold; text-align: center;
                    background-color: white; border-radius: 50%; width: 30px; 
                    height: 30px; line-height: 30px; border: 2px solid #6c757d;">
                    L</div>""")
            ).add_to(m)
    
    @staticmethod
    def _add_route_visualization(m: folium.Map, sequence: List[Stop], 
                                 hauling_loc: Tuple[float, float],
                                 landfill_locs: List[Tuple[float, float]],
                                 route_info: Dict,
                                 swing_decisions: Dict = None):
        # Draw main route lines (Haul → stops → Haul)
        locations = [hauling_loc]
        for stop in sequence:
            locations.append((stop.latitude, stop.longitude))
        locations.append(hauling_loc)
        
        # main_route = folium.PolyLine(
        #     locations=locations,
        #     color=RouteVisualizer.OPERATION_COLORS["MAIN_ROUTE"],
        #     weight=3,
        #     opacity=1,
        #     popup="Main Route",
        #     tooltip="Main Route"
        # ).add_to(m)
        
        # plugins.PolyLineTextPath(
        #     polyline=main_route,
        #     text='→',
        #     offset=20,
        #     repeat=True,
        #     attributes={'fill': RouteVisualizer.OPERATION_COLORS["MAIN_ROUTE"],
        #                 'font-size': '14px'}
        # ).add_to(m)
        
        # Add offset handling for close markers
        seen_coords = {}  # Use dictionary to count occurrences
        OFFSET_DISTANCE = 0.0002  # Approximately 11 meters
        locations_v = [hauling_loc]
        locations_v = [hauling_loc]
        for i, stop in enumerate(sequence):
            base_coord = (stop.latitude, stop.longitude)
            count = seen_coords.get(base_coord, 0)
            # if count > 0:
            #     # Calculate offset in a circular pattern
            #     angle = (2 * math.pi * int(stop.name.split()[-1])) / (int(stop.name.split()[-1]) + 1)
            #     lat_offset = OFFSET_DISTANCE * math.cos(angle)
            #     lng_offset = OFFSET_DISTANCE * math.sin(angle)
            #     marker_coord = (base_coord[0] + lat_offset, base_coord[1] + lng_offset)
            # else:
            #     marker_coord = base_coord
            angle = (2 * math.pi * int(stop.name.split()[-1])) / (int(stop.name.split()[-1]) + 1)
            lat_offset = OFFSET_DISTANCE * math.cos(angle)
            lng_offset = OFFSET_DISTANCE * math.sin(angle)
            marker_coord = (base_coord[0] + lat_offset, base_coord[1] + lng_offset)
            seen_coords[base_coord] = count + 1
            locations_v.append(marker_coord)
            is_swing = route_info.get(int(stop.id) - 1, 0)
            is_cand_swing = swing_decisions.get(int(stop.id) - 1, 0)
            color = RouteVisualizer.OPERATION_COLORS["SWG"] if is_cand_swing == 1 else RouteVisualizer.OPERATION_COLORS["DRT"]
            
            folium.Marker(
                marker_coord,
                # popup=f"""<div style='font-size: 14px'>
                #          <b>Stop {i+1}</b><br>
                #          Name: {stop.name}<br>
                #          Container: {stop.container_size}<br>
                #          Operation: {'SWING' if is_swing else 'DRT'}</div>""",
                popup=f"""<div style='font-size: 14px'>
                         <b>{i + 1}</b><br>
                         Name: {stop.name}<br>
                         Container: {stop.container_size}<br>
                         Operation: {'SWING' if is_cand_swing else 'DRT'}</div>""",
                icon=folium.DivIcon(html=f"""
                    <div style="font-family: courier new; color: {color}; 
                    font-size: 20px; font-weight: bold; text-align: center;
                    background-color: white; border-radius: 50%; width: 30px; 
                    height: 30px; line-height: 30px; border: 2px solid {color};">
                    {i + 1}</div>""")
            ).add_to(m)
        locations_v.append(hauling_loc)
        main_route = folium.PolyLine(
            locations=locations_v,
            color=RouteVisualizer.OPERATION_COLORS["MAIN_ROUTE"],
            weight=3,
            opacity=1,
            popup="Main Route",
            tooltip="Main Route"
        ).add_to(m)
        plugins.PolyLineTextPath(
            polyline=main_route,
            text='→',
            offset=20,
            repeat=True,
            attributes={'fill': RouteVisualizer.OPERATION_COLORS["MAIN_ROUTE"],
                        'font-size': '14px'}
        ).add_to(m)
        # Add landfill trip lines for non-swing stops (unchanged)
        for i, stop in enumerate(sequence):
            is_swing = route_info.get(int(stop.id) - 1, 0)
            if is_swing == 0:
                current_loc = (stop.latitude, stop.longitude)
                if len(landfill_locs) > 1:
                    distances = [haversine_distance(current_loc, lf) for lf in landfill_locs]
                    min_index = distances.index(min(distances))
                    chosen_landfill = landfill_locs[min_index]
                else:
                    chosen_landfill = landfill_locs[0]
                folium.PolyLine(
                    locations=[current_loc, chosen_landfill],
                    color=RouteVisualizer.OPERATION_COLORS["LANDFILL"],
                    weight=2,
                    opacity=0.7,
                    dash_array='10',
                    popup=f"Landfill Trip for Stop {i+1}",
                    tooltip=f"To Landfill"
                ).add_to(m)
                folium.PolyLine(
                    locations=[chosen_landfill, current_loc],
                    color=RouteVisualizer.OPERATION_COLORS["LANDFILL"],
                    weight=2,
                    opacity=0.7,
                    dash_array='10',
                    popup=f"Return from Landfill for Stop {i+1}",
                    tooltip=f"Return from Landfill"
                ).add_to(m)
    
    @staticmethod
    def _add_legend(m: folium.Map):
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; 
                    right: 50px; 
                    width: 200px;
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);
                    z-index: 1000;">
            <h4 style="margin-top: 0;">Route Legend</h4>
            <div style="margin-bottom: 8px;">
                <div style="display: inline-block; width: 20px; height: 3px; 
                     background-color: #007bff; margin-right: 5px;"></div>
                <span>Main Route →</span>
            </div>
            <div style="margin-bottom: 8px;">
                <div style="display: inline-block; width: 20px; height: 3px; 
                     background-color: #6c757d; border-style: dashed; margin-right: 5px;"></div>
                <span>Landfill Trip</span>
            </div>
            <div style="margin-bottom: 8px;">
                <div style="display: inline-block; width: 20px; height: 20px; 
                     border: 2px solid #28a745; border-radius: 50%; text-align: center; 
                     line-height: 16px; margin-right: 5px;">1</div>
                <span>SWING Stop</span>
            </div>
            <div style="margin-bottom: 8px;">
                <div style="display: inline-block; width: 20px; height: 20px; 
                     border: 2px solid #dc3545; border-radius: 50%; text-align: center; 
                     line-height: 16px; margin-right: 5px;">1</div>
                <span>DRT Stop</span>
            </div>
            <div style="margin-bottom: 8px;">
                <div style="display: inline-block; width: 20px; height: 20px; 
                     border: 2px solid #007bff; border-radius: 50%; text-align: center; 
                     line-height: 16px; margin-right: 5px;">H</div>
                <span>Hauling Facility</span>
            </div>
            <div style="margin-bottom: 8px;">
                <div style="display: inline-block; width: 20px; height: 20px; 
                     border: 2px solid #6c757d; border-radius: 50%; text-align: center; 
                     line-height: 16px; margin-right: 5px;">L</div>
                <span>Landfill</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

# ===============================
# API / Mathematical Helpers
# ===============================
def fetch_distance_matrix(locations):
    latitudes = [loc[0] for loc in locations]
    longitudes = [loc[1] for loc in locations]
    points = ";".join([f"{lon},{lat}" for lat, lon in zip(latitudes, longitudes)])
    url = "https://dev-gisweb.repsrv.com/rise/rest/services/Routing/NetworkAnalysis/NAServer/OriginDestinationCostMatrix/solveODCostMatrix"
    params = {
        "f": "json",
        "origins": points,
        "destinations": points,
        "MeasurementUnits": "Miles",
        "ImpedanceAttributeName": "TravelTime",
        "AccumulateAttributeNames": "TravelTime,Miles",
        "ReturnRoutes": "False",
        "ReturnStops": "False",
        "outputLines": "esriNAOutputLineNone",
        "spatialReference": '{"wkid": 4326}'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        od_cost_matrix = data.get("odCostMatrix", {})
        if not od_cost_matrix:
            raise ValueError("Missing 'odCostMatrix' in API response.")
        cost_attribute_names = od_cost_matrix.get("costAttributeNames", [])
        if "Miles" not in cost_attribute_names:
            raise ValueError("Missing 'Miles' attribute in cost matrix.")
        miles_index = cost_attribute_names.index("Miles")
        num_locations = len(locations)
        distance_matrix = np.zeros((num_locations, num_locations))
        for origin_key, destinations in od_cost_matrix.items():
            if origin_key.isdigit():
                origin_index = int(origin_key) - 1
                for dest_key, costs in destinations.items():
                    if dest_key.isdigit() and isinstance(costs, list):
                        dest_index = int(dest_key) - 1
                        if origin_index < num_locations and dest_index < num_locations:
                            distance_matrix[origin_index, dest_index] = costs[miles_index]
        return distance_matrix
    except Exception as e:
        print(f"Error fetching distance matrix: {e}")
        return np.zeros((len(locations), len(locations)))

def haversine_distance(coord1, coord2):
    R = 6371.0
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def travel_time_Google(lon1, lat1, lon2, lat2, API_KEY):
    gmaps = googlemaps.Client(key=API_KEY)
    origin = (lat1, lon1)
    destination = (lat2, lon2)
    departure_time = datetime.datetime.now()
    directions = gmaps.directions(origin, destination, mode="driving", departure_time=departure_time)
    distance = directions[0]['legs'][0]['distance']['text']
    time_value = np.round(directions[0]['legs'][0]['duration_in_traffic']['value'] / 60, 2)
    return distance, time_value

def construct_sequence(arcs):
    sequence = [arcs[0][0], arcs[0][1]]
    arcs = arcs[1:]
    while arcs:
        for i, (start, end) in enumerate(arcs):
            if sequence[-1] == start:
                sequence.append(end)
                arcs.pop(i)
                break
    return sequence

def calculate_distance_and_time_matrix(locations):
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    time_matrix = np.zeros((num_locations, num_locations))
    avg_speed_kmh = 40
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance = haversine_distance(locations[i], locations[j])
                distance_matrix[i, j] = distance
                time_matrix[i, j] = distance / avg_speed_kmh
    return distance_matrix, time_matrix

def TSP_solver(customers, cij):
    N = [0] + customers
    mdl = Model()
    xij = mdl.addVars(N, N, vtype=GRB.BINARY)
    u = mdl.addVars(N, vtype=GRB.INTEGER, lb=1)
    mdl.setObjective(quicksum(cij[i, j] * xij[i, j] for i in N for j in N if i != j), GRB.MINIMIZE)
    mdl.addConstrs(quicksum(xij[i, j] for i in N if i != j) == 1 for j in N)
    mdl.addConstrs(quicksum(xij[j, i] for i in N if i != j) == 1 for j in N)
    mdl.addConstrs(u[i] - u[j] + 1 <= (len(N) - 1) * (1 - xij[i, j])
                   for i in N for j in N if i != j and i >= 2 and j >= 2)
    mdl.Params.TimeLimit = 90
    mdl.setParam('OutputFlag', 0)
    mdl.optimize()
    arcs = []
    for i in N:
        for j in N:
            if i != j and xij[i, j].x > 0.5:
                arcs.append((i, j))
    sequence = construct_sequence(arcs)
    return sequence

# ===============================
# Route Optimization System & Stop Creation
# ===============================
class RouteOptimizationSystem:
    def __init__(self):
        self.output_dir = "route_optimization_output"
        self.ensure_output_directories()
        
    def ensure_output_directories(self):
        directories = [
            self.output_dir,
            f"{self.output_dir}/maps",
            f"{self.output_dir}/reports",
            f"{self.output_dir}/data"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def create_stop(self, 
                    phase: int,
                    stop_id: str,
                    latitude: float,
                    longitude: float,
                    container_size: str,
                    name: str,
                    can_swing: bool = True,
                    current_container: str = "",
                    operation_type: str = "DRT",
                    is_compactor: bool = False,
                    has_space_for_swing: bool = True,
                    delivery_container_size: str = ""
                   ) -> Stop:
        if phase == 1:
            return Stop(stop_id, latitude, longitude, container_size, name)
        elif phase == 2:
            return SwingEnabledStop(stop_id, latitude, longitude, container_size, name, can_swing, current_container or container_size)
        else:
            return EnhancedStop(stop_id, latitude, longitude, container_size, 
                                name, can_swing, current_container or container_size, 
                                operation_type, is_compactor, has_space_for_swing, delivery_container_size)

def create_default_stops(phase: int, land_fill_dic, DEFAULT_LOCATIONS, DEFAULT_CONTAINERS) -> List[Stop]:
    system = RouteOptimizationSystem()
    stops = []
    skip_count = 1 + len(land_fill_dic)
    for i, (lat, lon, name) in enumerate(DEFAULT_LOCATIONS[skip_count:]):
        stop = system.create_stop(
            phase=phase,
            stop_id=str(i+1),
            latitude=lat,
            longitude=lon,
            container_size=DEFAULT_CONTAINERS[name],
            name=name,
            can_swing=True,
            current_container=DEFAULT_CONTAINERS[name]
        )
        stops.append(stop)
    return stops

def create_route_stops(phase: int, sequence, land_fill_dic, stops, landfill_locs) -> List[Stop]:
    system = RouteOptimizationSystem()
    newSequence = []
    for i in sequence:
        if i == 0:
            # Haul Center
            continue
        elif i == 1:
            # Landfill
            continue
        else:
            # Service stops
            stop_index = i - 2
            if stop_index < 0 or stop_index >= len(stops):
                print(f"Warning: stop index {stop_index} out of range for stops.")
                continue
            stop_obj = stops[stop_index]

            stop = system.create_stop(
                phase=phase,
                stop_id=str(i+1),
                latitude=stop_obj.latitude,
                longitude=stop_obj.longitude,
                container_size=stop_obj.container_size,
                name=stop_obj.name,
                can_swing=True,
                current_container=stop_obj.container_size
            )
            newSequence.append(stop)
    return newSequence

async def generate_route_map(location_id: str):
    """
    Main function to optimize route, generate maps, save CSVs, and return full JSON result.
    """

    # --- Setup and Load Data ---
    SERVICE_TIME_PER_STOP = 10 / 60  # 10 minutes (in hours)
    location_id_for_name = location_id.replace("/", "_")
    route_location_id = location_id.rsplit("-", 1)[0]

    df_routes = pd.read_csv('uploaded_files/transformed_data_snowflk.csv')
    df_landfill = pd.read_csv('uploaded_files/landFill_Haul.csv')
    df_routes["Route #"] = df_routes["Route #"].astype(str)
    df_landfill["Cost"] = df_landfill["Cost/Ton"].str.split('/').str[0].astype(float)

    df_route = df_routes[df_routes['Route #'] == route_location_id].copy()
    if df_route.empty:
        raise ValueError(f"No route found for {route_location_id}")

    # --- Extract coordinates and addresses ---
    df_route['haul_center_address'] = df_route.apply(lambda row: f"{row['HF_ADDRESS_LINE1']}, {row['HF_ADDRESS_CITY']}, {row['HF_ADDRESS_STATE']}", axis=1)
    df_route['lf_address'] = df_route.apply(lambda row: f"{row['DF_ADDRESS_LINE1']}, {row['DF_ADDRESS_CITY']}, {row['DF_ADDRESS_STATE']}", axis=1)

    sorted_df = df_route.sort_values(by=['Route #', 'SEQUENCE'])
    service_coords = [(row['Latitude'], row['Longitude']) for _, row in sorted_df.iterrows()]
    midpoint = (df_route.iloc[0]['HL_Lat'], df_route.iloc[0]['HL_Longt'])
    landfill = (df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'])

    locations = [(midpoint[0], midpoint[1]), (landfill[0], landfill[1])] + service_coords

    # --- Create container/service data ---
    containers, swg, service_time, perm_notes = [-1, -1], [np.nan, np.nan], [], ["", ""]

    for idx in range(2, len(locations)):
        match = df_route[(df_route['Latitude'] == locations[idx][0]) & (df_route['Longitude'] == locations[idx][1])]
        if match.empty:
            continue
        containers.append(str(match.iloc[0]['CURRENT_CONTAINER_SIZE']))
        swg.append(1 if match.iloc[0]['SERVICE_TYPE_CD'] == 'SWG' else 0)
        service_time.append(float(match.iloc[0]['SERVICE_WINDOW_TIME']) / 60)
        perm_notes.append(match.iloc[0]['PERM_NOTES'])
    route_info_dict = {idx: swing for idx, swing in enumerate(swg)}
    # --- Calculate Distance/Time Matrices ---
    dist_matrix, time_matrix = calculate_distance_and_time_matrix(locations)

    # --- Split SWG and DRT Stops ---
    swg_stops = [idx for idx, s in enumerate(swg) if s == 1 and idx > 1]
    drt_stops = [idx for idx, s in enumerate(swg) if s == 0 and idx > 1]

    # --- Optimize Routes ---
    routeOptimizedNew = [0]
    if swg_stops:
        routeOptimizedNew += TSP_solver(swg_stops, time_matrix)
        routeOptimizedNew.append(1)  # After SWG, go to landfill

    if drt_stops:
        for drt in TSP_solver(drt_stops, time_matrix):
            routeOptimizedNew += [0, drt, 1]

    if routeOptimizedNew[-1] != 0:
        routeOptimizedNew.append(0)

    manual_route = create_manual_route(swg, len(service_coords), landfill_locs=1)

    # --- Calculate Driving Distance/Time ---
    total_service_time = sum(service_time)
    totalDrivingDistanceOptimal, totalDrivingTimeOptimal = calculate_route_cost(routeOptimizedNew, dist_matrix, time_matrix, total_service_time)
    totalDrivingDistanceManual, totalDrivingTimeManual = calculate_route_cost(manual_route, dist_matrix, time_matrix, total_service_time)

    # --- Prepare JSON Result ---
    percentage_drt = round(swg.count(0) / (len(swg) - 2) * 100, 2)
    percentage_swg = round(swg.count(1) / (len(swg) - 2) * 100, 2)
    routeIDOptimal_list = [get_stop_name(i, landfill_locs=1) for i in routeOptimizedNew]
    routeIDManual_list = [get_stop_name(i, landfill_locs=1) for i in manual_route]
    df_info = df_route.iloc[0]

    result_json = {
        "Route_ID": location_id,
        "Driving Time (min) Optimal": totalDrivingTimeOptimal,
        "Driving Distance (mile) Optimal": totalDrivingDistanceOptimal,
        "Driving Time (min.) Manual": totalDrivingTimeManual,
        "Driving Distance (mile) Manual": totalDrivingDistanceManual,
        "Percentage of DRT": percentage_drt,
        "Percentage of Swing": percentage_swg,
        "Number of Stops": len(swg) - 2,
        "Route Optimal": routeIDOptimal_list,
        "Route Manual": routeIDManual_list,
        "DATE": df_info["SERVICE_DATE"],
        "HF_DIVISION_NAME": df_info["HF_DIVISION_NAME"],
        "HF_SITE_NAME": df_info["HF_SITE_NAME"],
        "HF_ADDRESS_LINE1": df_info["HF_ADDRESS_LINE1"],
        "HF_ADDRESS_LINE2": df_info["HF_ADDRESS_LINE2"],
        "HF_ADDRESS_CITY": df_info["HF_ADDRESS_CITY"],
        "HF_ADDRESS_STATE": df_info["HF_ADDRESS_STATE"],
        "HF_ADDRESS_POSTAL_CODE": df_info["HF_ADDRESS_POSTAL_CODE"],
        "DF_FACILITY_NAME": df_info["DF_FACILITY_NAME"],
        "DF_ADDRESS_LINE1": df_info["DF_ADDRESS_LINE1"],
        "DF_ADDRESS_LINE2": df_info["DF_ADDRESS_LINE2"],
        "DF_ADDRESS_CITY": df_info["DF_ADDRESS_CITY"],
        "DF_ADDRESS_STATE": df_info["DF_ADDRESS_STATE"],
        "DF_ADDRESS_POSTAL_CODE": df_info["DF_ADDRESS_POSTAL_CODE"],
        "Time Benefit": totalDrivingTimeManual - totalDrivingTimeOptimal,
        "Distance Benefit": totalDrivingDistanceManual - totalDrivingDistanceOptimal,
        "Benefit": totalDrivingTimeManual > totalDrivingTimeOptimal
    }

    # --- Save CSV Results ---
    save_result_csv(result_json, location_id_for_name)
    save_sequence_csv(routeOptimizedNew, manual_route, time_matrix, service_time, perm_notes, location_id_for_name)

    # --- Save Maps ---
    DEFAULT_CONTAINERS = {}
    for i in range(1, len(service_coords) + 1):
        DEFAULT_CONTAINERS[f"Stop {i}"] = "30"
    stops11 = create_default_stops(1, {"LF1": landfill}, [(midpoint[0], midpoint[1], "Hauling"), (landfill[0], landfill[1], "LF")] + [(lat, lon, f"Stop {i+1}") for i, (lat, lon) in enumerate(service_coords)], DEFAULT_CONTAINERS)
    hauling_loc_coord = midpoint
    save_maps(location_id_for_name, stops11, routeOptimizedNew, manual_route, landfill, hauling_loc_coord, route_info_dict)

    return result_json

# === Helper Functions ===

def calculate_route_cost(route, dist_matrix, time_matrix, service_time_sum):
    distance, time = 0, 0
    for i in range(len(route)-1):
        distance += dist_matrix[route[i], route[i+1]]
        time += time_matrix[route[i], route[i+1]]
    return distance, time + service_time_sum

def create_manual_route(swg, num_stops, landfill_locs=1):
    route = [0]
    pointer = landfill_locs + 1
    while pointer <= landfill_locs + num_stops:
        route.append(pointer)
        if swg[pointer] == 0:
            route += [1, pointer]
        pointer += 1
    if route[-1] != 0:
        route.append(0)
    return route

def get_stop_name(idx, landfill_locs=1):
    if idx == 0:
        return "Haul"
    elif idx <= landfill_locs:
        return f"LF{idx}"
    else:
        return f"Stop {idx - landfill_locs}"

def save_result_csv(result_json, location_id_for_name):
    os.makedirs('services/route_optimization_output/IND_results', exist_ok=True)
    df_result = pd.DataFrame([result_json])
    df_result.to_csv(f'services/route_optimization_output/IND_results/IND_results{location_id_for_name}.csv', index=False)

def save_sequence_csv(optimal_route, manual_route, time_matrix, service_time, perm_notes, location_id_for_name):
    os.makedirs('services/route_optimization_output/Sequence', exist_ok=True)
    seq_rows = []

    def sequence_rows(route, label):
        for i in range(len(route) - 1):
            start, end = route[i], route[i+1]
            travel_time = time_matrix[start, end] * 60
            service_time_extra = 0
            if end > 1:
                idx = end - 2
                if idx < len(service_time):
                    service_time_extra = service_time[idx] * 60
            total_time = travel_time + service_time_extra
            seq_rows.append({
                "Route_ID": location_id_for_name,
                "Route_Type": label,
                "Segment": f"{get_stop_name(start)} -> {get_stop_name(end)}",
                "Time (min)": round(total_time, 2),
                "Distance (km)": round(0, 2),
                "Service Time": service_time_extra,
                "PERM_NOTES": perm_notes[start],
                "NOTE": perm_notes[start]
            })

    sequence_rows(optimal_route, "Optimal")
    sequence_rows(manual_route, "Manual")

    pd.DataFrame(seq_rows).to_csv(f'services/route_optimization_output/Sequence/sequence_row{location_id_for_name}.csv', index=False)

def save_maps(route_id, stops, optimal_route, manual_route, landfill, haul_center, route_info):
    os.makedirs('maps', exist_ok=True)
    optimal_seq = create_route_stops(1, [i for i in optimal_route if i > 1], {"LF1": landfill}, stops, [landfill])
    manual_seq = create_route_stops(1, [i for i in manual_route if i > 1], {"LF1": landfill}, stops, [landfill])

    RouteVisualizer.create_map(
        stops=stops,
        sequence=optimal_seq,
        landfill_locs=[landfill],
        route_info=route_info,
        hauling_loc=haul_center
    ).save(f"maps/optimal_map{route_id}.html")

    RouteVisualizer.create_map(
        stops=stops,
        sequence=manual_seq,
        landfill_locs=[landfill],
        hauling_loc=haul_center
    ).save(f"maps/manual_map{route_id}.html")

import asyncio

if __name__ == "__main__":
    location_id = '4319_6/20/2024-820'
    result = asyncio.run(generate_route_map(location_id))
    print(result)
