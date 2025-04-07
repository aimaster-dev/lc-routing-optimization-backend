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

def create_route_stops(phase: int, sequence, land_fill_dic, locations, landfill_locs) -> List[Stop]:
    system = RouteOptimizationSystem()
    newSequence = []
    for i in sequence:
        if i == 0:
            name = 'Haul'
        elif i <= len(landfill_locs):
            name = 'LF'
        else:
            name = f"Stop {i - len(landfill_locs)}"
        lat = locations[i][0]
        lon = locations[i][1]
        stop = system.create_stop(
            phase=phase,
            stop_id=str(i+1),
            latitude=lat,
            longitude=lon,
            container_size='30',
            name=name,
            can_swing=True,
            current_container='30'
        )
        newSequence.append(stop)
    return newSequence

# ===============================
# Main Function: generate_route_map
# ===============================
# This asynchronous function processes one route (identified by location_id).
# async def generate_route_map(location_id: str):
#     SERVICE_TIME_PER_STOP = 10/60  # minutes

#     filtered_df = pd.read_csv('uploaded_files/new_data_2025.csv')
#     filtered_df["Route #"] = filtered_df["Route #"].astype(str)
#     API_KEY = 'xxx'
#     ESRI = True
#     trial = True
#     dictionary_outcome = {}

#     land_fill_data_frame = pd.read_csv('uploaded_files/landFill_Haul.csv')
#     land_fill_data_frame["Cost"] = land_fill_data_frame["Cost/Ton"].str.split('/').str[0].astype(float)

#     df_route = filtered_df[filtered_df['Route #'] == location_id].copy()
#     df_route['haul_center_address'] = df_route.apply(
#         lambda row: f"{row['HF_ADDRESS_LINE1']}, {row['HF_ADDRESS_CITY']}, {row['HF_ADDRESS_STATE']}",
#         axis=1
#     )
#     df_route['lf_address'] = df_route.apply(
#         lambda row: f"{row['DF_ADDRESS_LINE1']}, {row['DF_ADDRESS_CITY']}, {row['DF_ADDRESS_STATE']}",
#         axis=1
#     )

#     sorted_df = df_route.sort_values(by=['Route #', 'SEQUENCE'])
#     result_dict = (sorted_df
#                    .groupby('Route #')
#                    .apply(lambda group: [(row['Latitude'], row['Longitude']) for _, row in group.iterrows()])
#                    .to_dict())
#     dummList = result_dict[location_id]
#     haul_center_address = df_route.iloc[0]['haul_center_address']
#     lf_address = df_route.iloc[0]['lf_address']
#     midpoint = (df_route.iloc[0]['HL_Lat'], df_route.iloc[0]['HL_Longt'])
#     landfill_locs = [(df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'])]

#     containers = [-1, -1]
#     swg = [np.nan, np.nan]
#     service_time = []
#     # Initialize PERM_NOTES with placeholder values for depot stops.
#     perm_notes = ["", ""]  
#     land_fill_dic = {}
#     swing_decisions = [False, False]

#     DEFAULT_LOCATIONS = [
#         (midpoint[0], midpoint[1], "Hauling"),
#         (df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'], "LF")
#     ]
#     locations = [
#         (midpoint[0], midpoint[1]),
#         (df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'])
#     ]
#     land_fill_dic['LF1'] = (df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'])

#     for loc in dummList:
#         locations.append(loc)

#     DEFAULT_CONTAINERS = {}
#     for ii in range(1, len(dummList) + 1):
#         DEFAULT_CONTAINERS[f"Stop {ii}"] = "30"
#         DEFAULT_LOCATIONS.append((locations[len(locations) - len(dummList) + ii - 1][0],
#                                   locations[len(locations) - len(dummList) + ii - 1][1],
#                                   f"Stop {ii}"))

#     stops11 = create_default_stops(1, land_fill_dic, DEFAULT_LOCATIONS, DEFAULT_CONTAINERS)

#     for i in range(len(swg), len(locations)):
#         match = df_route[(df_route['Latitude'] == locations[i][0]) & 
#                          (df_route['Longitude'] == locations[i][1])]
#         containers.append(str(match.iloc[0]['CURRENT_CONTAINER_SIZE']))
#         service_type = match.iloc[0]['SERVICE_TYPE_CD']
#         service_time.append(float(match.iloc[0]['SERVICE_WINDOW_TIME']) / 60)
#         # Append the PERM_NOTES value for this stop:
#         perm_notes.append(match.iloc[0]['PERM_NOTES'])
#         if service_type == 'SWG':
#             swg.append(1)
#             swing_decisions.append(True)
#         else:
#             swg.append(0)
#             swing_decisions.append(False)

#     # --- Simplified Manual Route-Building Logic ---
#     # Simply build the route in file order: depot then every stop then depot.
#     route = [0]
#     for pointer in range(len(land_fill_dic) + 1, len(containers)):
#         route.append(pointer)
#     route.append(0)
#     # --- End Simplified Manual Route-Building Logic ---

#     # --- Group stops for optimization ---
#     # Only group DRT stops (swg==0) for TSP optimization; SWG stops remain in file order.
#     fullDictionary = {}
#     time_windows_dictionary = {}
#     for idx, val in enumerate(containers):
#         if val == -1:
#             continue
#         if swg[idx] == 0:
#             fullDictionary.setdefault('Null', []).append(idx)
#         time_windows_dictionary[idx] = [0, 60*8]

#     # --- Compute distance and time matrices ---
#     dist_matrix, time_matrix = calculate_distance_and_time_matrix(locations)

#     # --- Simplified Optimal Route-Building Logic ---
#     # Use the manual route as base, and re-order only the DRT stops.
#     manual_route = route[:]  # file order
#     if 'Null' in fullDictionary:
#         optimized_drt = TSP_solver(fullDictionary['Null'], time_matrix)
#         routeOptimizedNew = []
#         drt_index = 0
#         for idx in manual_route:
#             if idx > len(landfill_locs) and swg[idx] == 0:
#                 if drt_index < len(optimized_drt):
#                     routeOptimizedNew.append(optimized_drt[drt_index])
#                     drt_index += 1
#                 else:
#                     routeOptimizedNew.append(idx)
#             else:
#                 routeOptimizedNew.append(idx)
#         if routeOptimizedNew[0] != 0:
#             routeOptimizedNew = [0] + routeOptimizedNew
#         if routeOptimizedNew[-1] != 0:
#             routeOptimizedNew.append(0)
#     else:
#         routeOptimizedNew = manual_route
#     # --- End Simplified Optimal Route-Building Logic ---

#     containerOnTruck = ['Null']
#     route_optimized_full_info = {}
#     runas_optimal = []
#     swing_decisions_opt = []
#     routeIDOptimal = []
#     for nu, i in enumerate(routeOptimizedNew):
#         if i == 0:
#             name = 'Haul'
#             runas_optimal.append('Haul')
#             swing_decisions_opt.append(False)
#         elif i <= len(landfill_locs):
#             name = 'LF1'
#             runas_optimal.append('LF1')
#             swing_decisions_opt.append(False)
#         else:
#             name = f"Stop {i - len(landfill_locs)}"
#             route_optimized_full_info[i] = swg[i]
#             if swg[i] == 0:
#                 runas_optimal.append('DRT')
#                 swing_decisions_opt.append(False)
#             else:
#                 runas_optimal.append('SWG')
#                 swing_decisions_opt.append(True)
#         routeIDOptimal.append(name)

#     # (Removed extra loop that appended additional stops from fullDictionary.)

#     distanceTot = 0
#     timeTot = 0
#     cummDist = [distanceTot]
#     cummTime = [timeTot]
#     for i in range(len(routeOptimizedNew)-1):
#         if trial:
#             distanceTot += dist_matrix[routeOptimizedNew[i], routeOptimizedNew[i+1]]
#             cummDist.append(distanceTot)
#             timeTot += time_matrix[routeOptimizedNew[i], routeOptimizedNew[i+1]]
#             cummTime.append(timeTot)
#         else:
#             distance_dummy, time_dummy = travel_time_Google(
#                 locations[routeOptimizedNew[i]][1], locations[routeOptimizedNew[i]][0],
#                 locations[routeOptimizedNew[i+1]][1], locations[routeOptimizedNew[i+1]][0],
#                 API_KEY
#             )
#             try:
#                 distance_dummy = float(distance_dummy.replace(' mi', ''))
#             except:
#                 distance_dummy = 1
#             distanceTot += distance_dummy
#             cummDist.append(distanceTot)
#             timeTot += time_dummy
#             cummTime.append(timeTot)
#     total_service_time = sum(service_time)
#     totalDrivingTimeOptimal = cummTime[-1] + total_service_time
#     totalDrivingDistanceOptimal = cummDist[-1]
#     distanceTotManual = 0
#     timeTotManual = 0
#     cummDistManual = [distanceTotManual]
#     cummTimeManual = [timeTotManual]
#     for i in range(len(route)-1):
#         if trial:
#             distanceTotManual += dist_matrix[route[i], route[i+1]]
#             cummDistManual.append(distanceTotManual)
#             timeTotManual += time_matrix[route[i], route[i+1]]
#             cummTimeManual.append(timeTotManual)
#         else:
#             distance_dummy, time_dummy = travel_time_Google(
#                 locations[route[i]][1], locations[route[i]][0],
#                 locations[route[i+1]][1], locations[route[i+1]][0],
#                 API_KEY
#             )
#             try:
#                 distance_dummy = float(distance_dummy.replace(' mi', ''))
#             except:
#                 distance_dummy = 1
#             distanceTotManual += distance_dummy
#             cummDistManual.append(distanceTotManual)
#             timeTotManual += time_dummy
#             cummTimeManual.append(timeTotManual)
#     total_service_time = sum(service_time)
#     totalDrivingTimeManual = cummTimeManual[-1] + total_service_time
#     totalDrivingDistanManual = cummDistManual[-1]

#     routeID = []
#     runas_manual = []
#     swing_decisionsManual = []
#     route_manual_full_info = {}
#     for nu, i in enumerate(route):
#         if i == 0:
#             runas_manual.append('Haul')
#             name = 'Haul'
#             swing_decisionsManual.append(False)
#         elif i <= len(landfill_locs):
#             name = 'LF1'
#             runas_manual.append('LF1')
#             swing_decisionsManual.append(False)
#         else:
#             name = f"Stop {i - len(landfill_locs)}"
#             route_manual_full_info[i] = swg[i]
#             if swg[i] == 1:
#                 runas_manual.append('SWG')
#                 swing_decisionsManual.append(True)
#             else:
#                 runas_manual.append('DRT')
#                 swing_decisionsManual.append(False)
#         routeID.append(name)

#     routeIDOptimal_list = []
#     for nu, i in enumerate(routeOptimizedNew):
#         if i == 0:
#             name = 'Haul'
#         elif i <= len(landfill_locs):
#             name = 'LF1'
#         else:
#             name = f"Stop {i - len(landfill_locs)}"
#         routeIDOptimal_list.append(name)

#     dictionary_outcome[location_id] = [
#         totalDrivingTimeOptimal, totalDrivingDistanceOptimal,
#         totalDrivingTimeManual, totalDrivingDistanManual,
#         np.round(swg.count(0)/(len(swg)-len(landfill_locs)-1)*100, 2),
#         np.round(swg.count(1)/(len(swg)-len(landfill_locs)-1)*100, 2),
#         len(swg)-len(landfill_locs)-1,
#         [routeIDOptimal_list, runas_optimal],
#         [routeID, runas_manual]
#     ]

#     newStops = []
#     for i in routeOptimizedNew:
#         if i <= len(landfill_locs):
#             continue
#         if i not in newStops:
#             newStops.append(i)
#     optimal_sequence = create_route_stops(1, newStops, land_fill_dic, locations, landfill_locs)
#     hauling_loc_coord = (df_route.iloc[0]['HL_Lat'], df_route.iloc[0]['HL_Longt'])
#     print("stop11 optimal: ", stops11)
#     print("optimal_sequence: ", optimal_sequence)
#     print("length of optimal_sequence: ", len(optimal_sequence))
#     route_map_optimal = RouteVisualizer.create_map(
#         stops=stops11,
#         sequence=optimal_sequence,
#         route_info=route_optimized_full_info,
#         swing_decisions=swing_decisionsManual,
#         landfill_locs=landfill_locs,
#         hauling_loc=hauling_loc_coord
#     )
#     new_s = location_id.replace("/", "_")
#     route_map_optimal.save(f"maps/optimal_map{new_s}.html")

#     newStopsManual = []
#     for i in route:
#         if i <= len(landfill_locs):
#             continue
#         if i not in newStopsManual:
#             newStopsManual.append(i)
#     manual_sequence = create_route_stops(1, newStopsManual, land_fill_dic, locations, landfill_locs)
#     print("stop11 optimal: ", stops11)
#     print("manual_sequence: ", manual_sequence)
#     print("length of manual_sequence: ", len(manual_sequence))
#     route_map_manual = RouteVisualizer.create_map(
#         stops=stops11,
#         sequence=manual_sequence,
#         route_info=route_manual_full_info,
#         swing_decisions=swing_decisionsManual,
#         landfill_locs=landfill_locs,
#         hauling_loc=hauling_loc_coord
#     )
#     route_map_manual.save(f"maps/manual_map{new_s}.html")
#     print(f"Maps saved for route {location_id}")

#     rows = []
#     for k, v in dictionary_outcome.items():
#         row = [k] + v[:7] + [v[7], v[8]]
#         rows.append(row)
#     columns = ["Route_ID", "Driving Time (min) Optimal", "Driving Distance (mile) Optimal", 
#                "Driving Time (min.) Manual", "Driving Distance (mile) Manual", 'Percentage of DRT', 
#                'Percentage of Swing', 'Number of Stops', 'Route Optimal', 'Route Manual']
    
#     df_results = pd.DataFrame(rows, columns=columns)
#     df_results.to_csv('IND_results.csv', index=False)

#     # === New Code: Save sequences row wise to "sequence_row.csv" ===
#     def get_stop_name(idx):
#         if idx == 0:
#             return "Haul"
#         elif idx == 1:
#             return "LF"
#         else:
#             return f"Stop {idx - len(landfill_locs)}"
#     updated_routeOptimizedNew = []
#     for i in range(len(routeOptimizedNew)):
#         start_idx = routeOptimizedNew[i]
#         if start_idx < 2 and i > 0 and i < len(routeOptimizedNew) - 1:
#             continue
#         updated_routeOptimizedNew.append(start_idx)
#     # optimal_segments = []
#     # print(updated_routeOptimizedNew)
#     # print("routeOptimzedNew: ", updated_routeOptimizedNew)
#     # for i in range(len(updated_routeOptimizedNew) - 1):
#     #     start_idx = updated_routeOptimizedNew[i]
#     #     print(start_idx)
#     #     end_idx = updated_routeOptimizedNew[i + 1]
#     #     segment_str = f"{get_stop_name(start_idx)} -> {get_stop_name(end_idx)}"
#     #     travel_time_min = time_matrix[start_idx, end_idx] * 60
#     #     additional_service_time = 0
#     #     if end_idx > 1:
#     #         service_index = end_idx - 2  
#     #         if service_index < len(service_time):
#     #             additional_service_time = service_time[service_index]
#     #     total_segment_time = travel_time_min + additional_service_time * 60
#     #     seg_distance = dist_matrix[start_idx, end_idx]
#     #     print(start_idx, end_idx)
#     #     print("distmatrix: ", dist_matrix)
#     #     optimal_segments.append({
#     #         "Route_ID": location_id,
#     #         "Route_Type": "Optimal",
#     #         "Segment": segment_str,
#     #         "Time (min)": round(total_segment_time, 2),
#     #         "Distance (km)": round(seg_distance, 2),
#     #         "PERM_NOTES": perm_notes[start_idx],   # Added PERM_NOTES from the starting stop
#     #         "Service_time": additional_service_time * 60
#     #     })
#     # print(optimal_segments)

#     optimal_segments = []
#     print("routeOptimzedNew: ", routeOptimizedNew)
#     for i in range(len(routeOptimizedNew) - 1):
#         start_idx = routeOptimizedNew[i]
#         print(start_idx)
#         end_idx = routeOptimizedNew[i + 1]
#         segment_str = f"{get_stop_name(start_idx)} -> {get_stop_name(end_idx)}"
#         travel_time_min = time_matrix[start_idx, end_idx] * 60
#         additional_service_time = 0
#         if end_idx > 1:
#             service_index = end_idx - 2  
#             if service_index < len(service_time):
#                 additional_service_time = service_time[service_index]
#         total_segment_time = travel_time_min + additional_service_time * 60
#         seg_distance = dist_matrix[start_idx, end_idx]
#         print(start_idx, end_idx)
#         print("distmatrix: ", dist_matrix)
#         optimal_segments.append({
#             "Route_ID": location_id,
#             "Route_Type": "Optimal",
#             "Segment": segment_str,
#             "Time (min)": round(total_segment_time, 2),
#             "Distance (km)": round(seg_distance, 2),
#             "PERM_NOTES": perm_notes[start_idx],   # Added PERM_NOTES from the starting stop
#             "Service_time": additional_service_time * 60
#         })
#     print(optimal_segments)

#     manual_segments = []
#     for i in range(len(route) - 1):
#         start_idx = route[i]
#         end_idx = route[i + 1]
#         segment_str = f"{get_stop_name(start_idx)} -> {get_stop_name(end_idx)}"
#         travel_time_min = time_matrix[start_idx, end_idx] * 60
#         additional_service_time = 0
#         if end_idx > 1:
#             service_index = end_idx - 2
#             if service_index < len(service_time):
#                 additional_service_time = service_time[service_index]
#         total_segment_time = travel_time_min + additional_service_time * 60
#         seg_distance = dist_matrix[start_idx, end_idx]
#         manual_segments.append({
#             "Route_ID": location_id,
#             "Route_Type": "Manual",
#             "Segment": segment_str,
#             "Time (min)": round(total_segment_time, 2),
#             "Distance (km)": round(seg_distance, 2),
#             "PERM_NOTES": perm_notes[start_idx],   # Added PERM_NOTES from the starting stop
#             "Service_time": additional_service_time * 60
#         })

#     all_segments = optimal_segments + manual_segments
#     seq_csv_file = "sequence_row.csv"
#     df_seq = pd.DataFrame(all_segments)
#     print(df_seq)
#     df_seq.to_csv(seq_csv_file, index=False)
#     return {location_id: (route_map_optimal, route_map_manual)}




async def generate_route_map(location_id: str):
    SERVICE_TIME_PER_STOP = 10/60  # minutes

    filtered_df = pd.read_csv('uploaded_files/transformed_data_snowflk.csv')
    filtered_df["Route #"] = filtered_df["Route #"].astype(str)
    API_KEY = 'xxx'
    ESRI = True
    trial = True
    dictionary_outcome = {}
    location_id_for_name = location_id.replace("/", "_")
    land_fill_data_frame = pd.read_csv('uploaded_files/landFill_Haul.csv')
    land_fill_data_frame["Cost"] = land_fill_data_frame["Cost/Ton"].str.split('/').str[0].astype(float)
    print("fefefefefe", location_id)
    route_location_id = location_id.rsplit("-", 1)[0]
    print("asdfasdfasdfasdf", route_location_id)
    df_route = filtered_df[filtered_df['Route #'] == route_location_id].copy()
    df_route['haul_center_address'] = df_route.apply(
        lambda row: f"{row['HF_ADDRESS_LINE1']}, {row['HF_ADDRESS_CITY']}, {row['HF_ADDRESS_STATE']}",
        axis=1
    )
    df_route['lf_address'] = df_route.apply(
        lambda row: f"{row['DF_ADDRESS_LINE1']}, {row['DF_ADDRESS_CITY']}, {row['DF_ADDRESS_STATE']}",
        axis=1
    )

    sorted_df = df_route.sort_values(by=['Route #', 'SEQUENCE'])
    result_dict = (sorted_df
                   .groupby('Route #')
                   .apply(lambda group: [(row['Latitude'], row['Longitude']) for _, row in group.iterrows()])
                   .to_dict())
    print(result_dict)
    dummList = result_dict[route_location_id]
    haul_center_address = df_route.iloc[0]['haul_center_address']
    lf_address = df_route.iloc[0]['lf_address']
    midpoint = (df_route.iloc[0]['HL_Lat'], df_route.iloc[0]['HL_Longt'])
    landfill_locs = [(df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'])]

    containers = [-1, -1]
    swg = [np.nan, np.nan]
    service_time = []
    perm_notes = ["", ""]
    land_fill_dic = {}
    swing_decisions = [False, False]

    DEFAULT_LOCATIONS = [
        (midpoint[0], midpoint[1], "Hauling"),
        (df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'], "LF")
    ]
    locations = [
        (midpoint[0], midpoint[1]),
        (df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'])
    ]
    land_fill_dic['LF1'] = (df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'])

    for loc in dummList:
        locations.append(loc)

    DEFAULT_CONTAINERS = {}
    for ii in range(1, len(dummList) + 1):
        DEFAULT_CONTAINERS[f"Stop {ii}"] = "30"
        DEFAULT_LOCATIONS.append((locations[len(locations) - len(dummList) + ii - 1][0],
                                  locations[len(locations) - len(dummList) + ii - 1][1],
                                  f"Stop {ii}"))

    stops11 = create_default_stops(1, land_fill_dic, DEFAULT_LOCATIONS, DEFAULT_CONTAINERS)

    for i in range(len(swg), len(locations)):
        match = df_route[(df_route['Latitude'] == locations[i][0]) & 
                         (df_route['Longitude'] == locations[i][1])]
        containers.append(str(match.iloc[0]['CURRENT_CONTAINER_SIZE']))
        service_type = match.iloc[0]['SERVICE_TYPE_CD']
        service_time.append(float(match.iloc[0]['SERVICE_WINDOW_TIME'])/60) 
        perm_notes.append(match.iloc[0]['PERM_NOTES'])
        if service_type == 'SWG':
            swg.append(1)
            swing_decisions.append(True)
        else:
            swg.append(0)
            swing_decisions.append(False)

    # Create a copy of the original service types for manual route visualization
    original_swg = copy.deepcopy(swg)
    original_swing_decisions = copy.deepcopy(swing_decisions)
    
    # Create an optimized version where DRT > 2 min becomes SWG
    optimized_swg = copy.deepcopy(swg)
    optimized_swing_decisions = copy.deepcopy(swing_decisions)
    
    # Convert DRT stops with service time > 2 minutes to SWG for optimization
    for i in range(len(swg)):
        if i >= 2:  # Skip depot entries
            idx = i - 2
            if idx < len(service_time) and swg[i] == 0 and service_time[idx] > 8 / 60:
                optimized_swg[i] = 1
                optimized_swing_decisions[i] = True

    # Manual route construction - using original service types
    route = [0]
    pointer = len(land_fill_dic) + 1
    while pointer < len(containers):
        route.append(pointer)
        # For DRT, we need to go to landfill after the stop
        if original_swg[pointer] == 0:
            route.append(1)  # Go to landfill
            route.append(pointer)
            pointer += 1
            # if pointer + 1 < len(containers) and original_swg[pointer+1] == 1:
            #     route.append(0)
            # pointer += 1
        # For SWG, we can proceed directly to the next stop without visiting landfill
        else:
            pointer += 1
            # If this is the last stop or next stop is DRT, go to landfill
            # if pointer >= len(containers) - 1 or original_swg[pointer+1] == 0:
            #     # route.append(1)  # Go to landfill
            #     if pointer < len(containers) - 1:
            #         pointer += 1
            #         continue
            #     else:
            #         route.append(0)
            #         break
            # # If next stop is SWG but different container, go to landfill
            # elif containers[pointer+1] != containers[pointer]:
            #     # route.append(1)  # Go to landfill
            #     pointer += 1
            #     continue
            # # If next stop is SWG with same container, proceed directly
            # else:
            #     pointer += 1
            #     continue
    if route[-1] != 0:
        route.append(0)

    # Create dictionaries for optimization based on the modified service types
    fullDictionary = {}
    time_windows_dictionary = {}
    for idx, val in enumerate(containers):
        if val == -1:
            continue
        if optimized_swg[idx] == 1:
            fullDictionary.setdefault('Null', []).append(idx)
        else:
            fullDictionary.setdefault('Null', []).append(idx)
        time_windows_dictionary[idx] = [0, 60*8]

    routeOptimizedNew = [0]
    arrival_time = 0
    dist_matrix, time_matrix = calculate_distance_and_time_matrix(locations)
    
    # Handle DRT stops (Direct to Landfill)
    if 'Null' in fullDictionary:
        sequence = TSP_solver(fullDictionary['Null'], time_matrix)
        routeOptimizedNew = [0]
        for i in range(1, len(sequence)-1):
            routeOptimizedNew.append(sequence[i])
            if i == 1:
                arrival_time += time_matrix[0, sequence[i]]
            else:
                arrival_time += time_matrix[sequence[i-1], sequence[i]]
            if optimized_swg[sequence[i]] == 0:
                routeOptimizedNew.append(1)
            # routeOptimizedNew.append(1)  # Go to landfill
                arrival_time += time_matrix[sequence[i], 1]
                routeOptimizedNew.append(sequence[i])
                arrival_time += time_matrix[1, sequence[i]]
        routeOptimizedNew.append(0)
        arrival_time += time_matrix[sequence[i], 0]
        del fullDictionary['Null']

    # Handle SWG stops (Swing) - allow direct progression between stops
    containerOnTruck = ['Null']
    current_pos = 0  # Start at hauling location
    
    # Process each container type for SWG stops
    for key in fullDictionary.keys():
        if len(fullDictionary[key]) == 0:
            continue
            
        # Find closest stop to current position
        firstCustomer = -1
        min_distance = np.inf
        for i in fullDictionary[key]:
            if time_matrix[current_pos, i] < min_distance:
                min_distance = time_matrix[current_pos, i]
                firstCustomer = i
                
        routeOptimizedNew.append(firstCustomer)
        arrival_time += time_matrix[current_pos, firstCustomer]
        current_pos = firstCustomer
        fullDictionary[key].remove(firstCustomer)
        
        # Process remaining stops for this container
        while len(fullDictionary[key]):
            # Find next closest stop
            nextCustomer = -1
            min_distance = np.inf
            for k in fullDictionary[key]:
                if time_matrix[current_pos, k] < min_distance:
                    min_distance = time_matrix[current_pos, k]
                    nextCustomer = k
                    
            routeOptimizedNew.append(nextCustomer)
            arrival_time += time_matrix[current_pos, nextCustomer]
            current_pos = nextCustomer
            fullDictionary[key].remove(nextCustomer)
            
        # After all stops for this container, go to landfill
        routeOptimizedNew.append(1)  # Go to landfill
        arrival_time += time_matrix[current_pos, 1]
        current_pos = 1
        
    # Return to hauling location at the end
    if routeOptimizedNew[-1] != 0:
        routeOptimizedNew.append(0)
        arrival_time += time_matrix[current_pos, 0]

    containerOnTruck = ['Null']
    route_optimized_full_info = {}
    runas_optimal = []
    swing_decisions_opt = []
    routeIDOptimal = []
    for nu, i in enumerate(routeOptimizedNew):
        if i == 0:
            name = 'Haul'
            runas_optimal.append('Haul')
            swing_decisions_opt.append(False)
        elif i <= len(landfill_locs):
            name = 'LF1'
            runas_optimal.append('LF1')
            swing_decisions_opt.append(False)
        else:
            name = f"Stop {i - len(landfill_locs)}"
            # Use the optimized service types for the optimal route
            route_optimized_full_info[i] = optimized_swg[i]
            if optimized_swg[i] == 0:
                runas_optimal.append('DRT')
                swing_decisions_opt.append(False)
            else:
                runas_optimal.append('SWG')
                swing_decisions_opt.append(True)
        routeIDOptimal.append(name)

    distanceTot = 0
    timeTot = 0
    cummDist = [distanceTot]
    cummTime = [timeTot]
    for i in range(len(routeOptimizedNew)-1):
        if trial:
            distanceTot += dist_matrix[routeOptimizedNew[i], routeOptimizedNew[i+1]]
            cummDist.append(distanceTot)
            timeTot += time_matrix[routeOptimizedNew[i], routeOptimizedNew[i+1]]
            cummTime.append(timeTot)
        else:
            distance_dummy, time_dummy = travel_time_Google(
                locations[routeOptimizedNew[i]][1], locations[routeOptimizedNew[i]][0],
                locations[routeOptimizedNew[i+1]][1], locations[routeOptimizedNew[i+1]][0],
                API_KEY
            )
            try:
                distance_dummy = float(distance_dummy.replace(' mi', ''))
            except:
                distance_dummy = 1
            distanceTot += distance_dummy
            cummDist.append(distanceTot)
            timeTot += time_dummy
            cummTime.append(timeTot)

    total_service_time = sum(service_time)

    totalDrivingTimeOptimal = cummTime[-1] + total_service_time
    totalDrivingDistanceOptimal = cummDist[-1]

    distanceTotManual = 0
    timeTotManual = 0
    cummDistManual = [distanceTotManual]
    cummTimeManual = [timeTotManual]
    for i in range(len(route)-1):
        if trial:
            distanceTotManual += dist_matrix[route[i], route[i+1]]
            cummDistManual.append(distanceTotManual)
            timeTotManual += time_matrix[route[i], route[i+1]]
            cummTimeManual.append(timeTotManual)
        else:
            distance_dummy, time_dummy = travel_time_Google(
                locations[route[i]][1], locations[route[i]][0],
                locations[route[i+1]][1], locations[route[i+1]][0],
                API_KEY
            )
            try:
                distance_dummy = float(distance_dummy.replace(' mi', ''))
            except:
                distance_dummy = 1
            distanceTotManual += distance_dummy
            cummDistManual.append(distanceTotManual)
            timeTotManual += time_dummy
            cummTimeManual.append(timeTotManual)
    
    totalDrivingTimeManual = cummTimeManual[-1] + total_service_time
    totalDrivingDistanManual = cummDistManual[-1]

    routeID = []
    runas_manual = []
    swing_decisionsManual = []
    route_manual_full_info = {}
    for nu, i in enumerate(route):
        if i == 0:
            runas_manual.append('Haul')
            name = 'Haul'
            swing_decisionsManual.append(False)
        elif i <= len(landfill_locs):
            name = 'LF1'
            runas_manual.append('LF1')
            swing_decisionsManual.append(False)
        else:
            name = f"Stop {i - len(landfill_locs)}"
            # Use original service types for manual route
            route_manual_full_info[i] = original_swg[i]
            if original_swg[i] == 1:
                runas_manual.append('SWG')
                swing_decisionsManual.append(True)
            else:
                runas_manual.append('DRT')
                swing_decisionsManual.append(False)
        routeID.append(name)

    routeIDOptimal_list = []
    for nu, i in enumerate(routeOptimizedNew):
        if i == 0:
            name = 'Haul'
        elif i <= len(landfill_locs):
            name = 'LF1'
        else:
            name = f"Stop {i - len(landfill_locs)}"
        routeIDOptimal_list.append(name)

    dictionary_outcome[location_id] = [
        totalDrivingTimeOptimal, totalDrivingDistanceOptimal,
        totalDrivingTimeManual, totalDrivingDistanManual,
        np.round(swg.count(0)/(len(swg)-len(landfill_locs)-1)*100, 2),
        np.round(swg.count(1)/(len(swg)-len(landfill_locs)-1)*100, 2),
        len(swg)-len(landfill_locs)-1,
        [routeIDOptimal_list, runas_optimal],
        [routeID, runas_manual]
    ]

    newStops = []
    for i in routeOptimizedNew:
        if i <= len(landfill_locs):
            continue
        if i not in newStops:
            newStops.append(i)
    optimal_sequence = create_route_stops(1, newStops, land_fill_dic, locations, landfill_locs)
    # print("optimal_sequence", optimal_sequence)
    # print("stop11", stops11)
    # print("route_optimized_full_info", route_optimized_full_info)
    # print("swing_decisions_opt", swing_decisions_opt)
    # print("swing_decisionsManual", swing_decisionsManual)
    hauling_loc_coord = (df_route.iloc[0]['HL_Lat'], df_route.iloc[0]['HL_Longt'])
    route_map_optimal = RouteVisualizer.create_map(
        stops=stops11,
        sequence=optimal_sequence,
        route_info=route_optimized_full_info,
        swing_decisions=route_optimized_full_info,
        landfill_locs=landfill_locs,
        hauling_loc=hauling_loc_coord
    )
    new_s = location_id.replace("/", "_")
    route_map_optimal.save(f"maps/optimal_map{new_s}.html")

    newStopsManual = []
    for i in route:
        if i <= len(landfill_locs):
            continue
        if i not in newStopsManual:
            newStopsManual.append(i)
    manual_sequence = create_route_stops(1, newStopsManual, land_fill_dic, locations, landfill_locs)
    route_map_manual = RouteVisualizer.create_map(
        stops=stops11,
        sequence=manual_sequence,
        route_info=route_manual_full_info,
        swing_decisions=route_manual_full_info,
        landfill_locs=landfill_locs,
        hauling_loc=hauling_loc_coord
    )
    route_map_manual.save(f"maps/manual_map{new_s}.html")
    # print(f"Maps saved for route {location_id}")

    rows = []
    for k, v in dictionary_outcome.items():
        row = [k] + v[:7] + [v[7], v[8]]
        rows.append(row)
    columns = ["Route_ID", "Driving Time (min) Optimal", "Driving Distance (mile) Optimal", 
            "Driving Time (min.) Manual", "Driving Distance (mile) Manual", 'Percentage of DRT', 
            'Percentage of Swing', 'Number of Stops', 'Route Optimal', 'Route Manual']
    
    df_results = pd.DataFrame(rows, columns=columns)
    # print(df_results)
    # df_results.to_csv('IND_results.csv', index=False)

    # === New Code: Save sequences row wise to "sequence_row.csv" ===
    def get_stop_name(idx):
        if idx == 0:
            return "Haul"
        elif idx == 1:
            return "LF"
        else:
        # For stops beyond the depots, subtract the number of depot entries (assumed to be len(landfill_locs))
        # For example, if landfill_locs has one element, then index 2 becomes "Stop 1", index 3 becomes "Stop 2", etc.
            return f"Stop {idx - len(landfill_locs)}"

    # Process the optimal route segments
    optimal_segments = []
    # print(routeOptimizedNew)
    sum_optimal = 0
    for i in range(len(routeOptimizedNew) - 1):
        start_idx = routeOptimizedNew[i]
        end_idx = routeOptimizedNew[i + 1]
        segment_str = f"{get_stop_name(start_idx)} -> {get_stop_name(end_idx)}"
        
        # Calculate travel time in minutes (time_matrix is in hours)
        travel_time_min = time_matrix[start_idx, end_idx] * 60
        
        # Add the service time at the destination stop if it is not a depot
        additional_service_time = 0
        if end_idx > 1:
            # Map the route index to the service_time list index (assuming stops start at index 2)
            service_index = end_idx - 2  
            if service_index < len(service_time):
                additional_service_time = service_time[service_index]
        
        total_segment_time = travel_time_min + additional_service_time * 60
        seg_distance = dist_matrix[start_idx, end_idx]  # Assumes distance is in km
        if isinstance(perm_notes[start_idx], str):
            time_match = re.search(r'(\d{1,2}(:\d{2})?[APM]{2})\b(?!\s*\.)', perm_notes[start_idx])
            if time_match:
                service_time_note = time_match.group(0)
            else:
                service_time_note = ""
        else:
            service_time_note = ""
        print("route_optimal_timing_calc", travel_time_min, additional_service_time)
        optimal_segments.append({
            "Route_ID": location_id,
            "Route_Type": "Optimal",
            "Segment": segment_str,
            "Time (min)": round(total_segment_time, 2),
            "Distance (km)": round(seg_distance, 2),
            "Service Time": additional_service_time * 60,
            "PERM_NOTES": service_time_note,
            "NOTE": perm_notes[start_idx]
        })
        sum_optimal += total_segment_time

    # Process the manual route segments
    # print(route)
    manual_segments = []
    sum_manual = 0
    for i in range(len(route) - 1):
        start_idx = route[i]
        end_idx = route[i + 1]
        segment_str = f"{get_stop_name(start_idx)} -> {get_stop_name(end_idx)}"
        
        travel_time_min = time_matrix[start_idx, end_idx] * 60
        
        additional_service_time = 0
        if end_idx > 1:
            service_index = end_idx - 2
            if service_index < len(service_time):
                additional_service_time = service_time[service_index]
        
        total_segment_time = travel_time_min + additional_service_time * 60
        seg_distance = dist_matrix[start_idx, end_idx]
        if isinstance(perm_notes[start_idx], str):
            time_match = re.search(r'(\d{1,2}(:\d{2})?[APM]{2})\b(?!\s*\.)', perm_notes[start_idx])
            if time_match:
                service_time_note = time_match.group(0)
            else:
                service_time_note = ""
        else:
            service_time_note = ""
        print("route_manual_timing_calc", travel_time_min, additional_service_time)
        manual_segments.append({
            "Route_ID": location_id,
            "Route_Type": "Manual",
            "Segment": segment_str,
            "Time (min)": round(total_segment_time, 2),
            "Distance (km)": round(seg_distance, 2),
            "Service Time": additional_service_time * 60,
            "PERM_NOTES": service_time_note, 
            "NOTE": perm_notes[start_idx]
        })
        sum_manual += total_segment_time

    # Combine and save the segment details to a CSV file
    all_segments = optimal_segments + manual_segments
    seq_csv_file = f"services/route_optimization_output/Sequence/sequence_row{location_id_for_name}.csv"
    # seq_csv_file = f"sequence_row{location_id_for_name}.csv"
    df_seq = pd.DataFrame(all_segments)
    df_seq.to_csv("sequence_row.csv", index=False)
    os.makedirs("services/route_optimization_output/Sequence/", exist_ok=True)
    try:
        df_seq.to_csv(seq_csv_file, index=False)
    except Exception as e:
        print(f"Failed to write CSV: {e}")
        pass
    # print(sum_manual, sum_optimal)
    df_results["Driving Time (min) Optimal"] = [sum_optimal / 60]
    df_results["Driving Time (min.) Manual"] = [sum_manual / 60]
    df_results.to_csv(f'IND_results.csv', index=False)
    try:
        df_results.to_csv(f'services/route_optimization_output/IND_results/IND_results{location_id_for_name}.csv', index=False)
    except Exception as e:
        print(f"Failed to write CSV: {e}")
        pass
    # df_results.to_csv(f'IND_results{location_id_for_name}.csv', index=False)

    columns = ["Route_ID", "Driving Time (min) Optimal", "Driving Distance (mile) Optimal", 
            "Driving Time (min.) Manual", "Driving Distance (mile) Manual", 'Percentage of DRT', 
            'Percentage of Swing', 'Number of Stops', 'Route Optimal', 'Route Manual']
    df_route_info = filtered_df[filtered_df['Route #'] == route_location_id].copy().iloc[0]
    print("old", df_results["Driving Time (min.) Manual"])
    return {
        "Route_ID": location_id,
        "Driving Time (min) Optimal": df_results["Driving Time (min) Optimal"],
        "Driving Distance (mile) Optimal": df_results["Driving Distance (mile) Optimal"],
        "Driving Time (min) Manual": df_results["Driving Time (min.) Manual"],
        "Driving Distance (mile) Manual": df_results["Driving Distance (mile) Manual"],
        "Percentage of DRT": df_results["Percentage of DRT"],
        "Percentage of Swing": df_results["Percentage of Swing"],
        "Number of Stops": df_results["Number of Stops"],
        "Route Optimal": df_results["Number of Stops"],
        "Route Manual": df_results["Number of Stops"],
        "DATE": df_route_info["SERVICE_DATE"],
        "HF_DIVISION_NAME": df_route_info["HF_DIVISION_NAME"],
        "HF_SITE_NAME": df_route_info["HF_SITE_NAME"],
        "HF_ADDRESS_LINE1": df_route_info["HF_ADDRESS_LINE1"],
        "HF_ADDRESS_LINE2": df_route_info["HF_ADDRESS_LINE2"],
        "HF_ADDRESS_CITY": df_route_info["HF_ADDRESS_CITY"],
        "HF_ADDRESS_STATE": df_route_info["HF_ADDRESS_STATE"],
        "HF_ADDRESS_POSTAL_CODE": df_route_info["HF_ADDRESS_POSTAL_CODE"],
        "DF_FACILITY_NAME": df_route_info["DF_FACILITY_NAME"],
        "DF_ADDRESS_LINE1": df_route_info["DF_ADDRESS_LINE1"],
        "DF_ADDRESS_LINE2": df_route_info["DF_ADDRESS_LINE2"],
        "DF_ADDRESS_CITY": df_route_info["DF_ADDRESS_CITY"],
        "DF_ADDRESS_STATE": df_route_info["DF_ADDRESS_STATE"],
        "DF_ADDRESS_POSTAL_CODE": df_route_info["DF_ADDRESS_POSTAL_CODE"],
        "Time Benefit": float(df_results["Driving Time (min.) Manual"].iloc[0]) - float(df_results["Driving Time (min) Optimal"].iloc[0]),
        "Distance Benefit": float(df_results["Driving Distance (mile) Manual"].iloc[0]) - float(df_results["Driving Distance (mile) Optimal"].iloc[0]),
        "Benefit": float(df_results["Driving Time (min.) Manual"].iloc[0]) > float(df_results["Driving Time (min) Optimal"].iloc[0])
    }