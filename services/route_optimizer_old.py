import pandas as pd
import numpy as np
import math
import os
import folium
from folium import plugins
from dataclasses import dataclass
from typing import List, Tuple, Dict
import requests
import asyncio
from gurobipy import Model, GRB, quicksum

# Ensure folders exist
os.makedirs("maps", exist_ok=True)
os.makedirs('services/route_optimization_output/IND_results', exist_ok=True)
os.makedirs('services/route_optimization_output/Sequence', exist_ok=True)

# === Data Classes ===
@dataclass
class Stop:
    id: str
    latitude: float
    longitude: float
    container_size: str
    name: str
    can_swing: bool = False
    current_container: str = ""
    operation_type: str = "DRT"

# === Visualizer ===
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
        stops,
        sequence,
        route_info,
        swing_decisions=None,
        landfill_locs=None,
        hauling_loc=None
    ) -> folium.Map:
        if hauling_loc is None:
            hauling_loc = (41.655032, -86.0097)
        if landfill_locs is None:
            landfill_locs = [(33.4353, -112.0065)]

        center_lat = (hauling_loc[0] + landfill_locs[0][0]) / 2
        center_lon = (hauling_loc[1] + landfill_locs[0][1]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

        RouteVisualizer._add_facility_markers(m, hauling_loc, landfill_locs)
        RouteVisualizer._add_route_visualization(m, sequence, hauling_loc, landfill_locs, route_info, swing_decisions)
        RouteVisualizer._add_legend(m)

        return m

    @staticmethod
    def _add_facility_markers(m, hauling_loc, landfill_locs):
        # Hauling center marker
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

        # Landfill markers
        for i, landfill_loc in enumerate(landfill_locs):
            folium.Marker(
                landfill_loc,
                popup=f"Landfill {i+1}",
                icon=folium.DivIcon(html=f"""\
                    <div style="font-family: courier new; color: #6c757d;
                    font-size: 24px; font-weight: bold; text-align: center;
                    background-color: white; border-radius: 50%; width: 30px;
                    height: 30px; line-height: 30px; border: 2px solid #6c757d;">
                    L</div>""")
            ).add_to(m)

    @staticmethod
    def _add_route_visualization(m, sequence, hauling_loc, landfill_locs, route_info, swing_decisions):
        OFFSET_DISTANCE = 0.0002  # Small offset for visualization
        
        # --- Prepare offset points to prevent overlapping ---
        shifted_points = []
        shifted_coords_for_markers = []  # save shifted marker coords

        for idx, stop in enumerate(sequence):
            base_lat, base_lon = stop.latitude, stop.longitude
            angle = (2 * math.pi * idx) / len(sequence)  # Spread points around circle
            lat_offset = OFFSET_DISTANCE * math.cos(angle)
            lon_offset = OFFSET_DISTANCE * math.sin(angle)
            shifted_lat = base_lat + lat_offset
            shifted_lon = base_lon + lon_offset
            shifted_points.append((shifted_lat, shifted_lon))
            shifted_coords_for_markers.append((shifted_lat, shifted_lon))  # save for marker use

        # --- Build full path with Hauling start and end ---
        points = [hauling_loc] + shifted_points + [hauling_loc]

        # --- Draw connecting lines ---
        for i in range(len(points) - 1):
            folium.PolyLine(
                locations=[points[i], points[i+1]],
                color=RouteVisualizer.OPERATION_COLORS["MAIN_ROUTE"],
                weight=3,
                opacity=1,
                popup="Main Route",
                tooltip="Main Route"
            ).add_to(m)

        # --- Add directional arrows ➔ ---
        for i in range(len(points) - 1):
            segment = folium.PolyLine(
                locations=[points[i], points[i+1]],
                color=RouteVisualizer.OPERATION_COLORS["MAIN_ROUTE"],
                weight=3,
                opacity=0  # transparent line for arrows
            ).add_to(m)

            plugins.PolyLineTextPath(
                segment,
                '➔',
                repeat=True,
                offset=20,
                attributes={'fill': RouteVisualizer.OPERATION_COLORS["MAIN_ROUTE"], 'font-size': '10px'}
            ).add_to(m)
        
        visited = set()  # track unique stops
        service_idx = 1  # real numbering for real stops (starts at 1)
        # --- Draw stop markers exactly at shifted points ---
        for idx, stop in enumerate(sequence):
            marker_coord = shifted_coords_for_markers[idx]
            stop_id = int(stop.id) - 1
            is_swing = route_info.get(stop_id, 0)

            # Only number if not visited yet
            if stop_id not in visited:
                label_text = str(service_idx)
                visited.add(stop_id)
                service_idx += 1
            else:
                continue
                # label_text = ""  # repeated stop: no number shown

            color = RouteVisualizer.OPERATION_COLORS["SWG"] if is_swing == 1 else RouteVisualizer.OPERATION_COLORS["DRT"]

            folium.Marker(
                marker_coord,
                popup=f"""<div style='font-size: 14px'>
                        <b>Stop</b><br>
                        Name: {stop.name}<br>
                        Container: {stop.container_size}<br>
                        Operation: {'SWING' if is_swing == 1 else 'DRT'}</div>""",
                icon=folium.DivIcon(html=f"""
                    <div style="font-family: courier new; color: {color};
                    font-size: 20px; font-weight: bold; text-align: center;
                    background-color: white; border-radius: 50%; width: 30px;
                    height: 30px; line-height: 30px; border: 2px solid {color};">
                    {label_text}</div>""")
            ).add_to(m)

        # --- Landfill trips (optional) still drawn based on real lat/lon ---
        for idx, stop in enumerate(sequence):
            stop_id = int(stop.id) - 1
            is_swing = route_info.get(stop_id, 0)

            current_loc = (stop.latitude, stop.longitude)  # Keep real position for landfill trips!

            if is_swing == 1:  # SWG stops
                if idx == len(sequence) - 1 or route_info.get(int(sequence[idx+1].id)-1, 0) == 0:
                    chosen_landfill = min(landfill_locs, key=lambda lf: haversine_distance(current_loc, lf))
                    folium.PolyLine(
                        locations=[current_loc, chosen_landfill],
                        color=RouteVisualizer.OPERATION_COLORS["LANDFILL"],
                        weight=2,
                        opacity=0.7,
                        dash_array='5,10',
                        popup=f"Landfill trip after SWG Stop {idx+1}",
                        tooltip="Landfill Trip"
                    ).add_to(m)

                    folium.PolyLine(
                        locations=[chosen_landfill, hauling_loc],
                        color="#28a745",
                        weight=3,
                        opacity=1,
                        popup="Return to Haul after SWG",
                        tooltip="Return to Haul"
                    ).add_to(m)

            elif is_swing == 0:  # DRT stops
                chosen_landfill = min(landfill_locs, key=lambda lf: haversine_distance(current_loc, lf))
                folium.PolyLine(
                    locations=[current_loc, chosen_landfill],
                    color=RouteVisualizer.OPERATION_COLORS["LANDFILL"],
                    weight=2,
                    opacity=0.7,
                    dash_array='5,10',
                    popup=f"Landfill trip after DRT Stop {idx+1}",
                    tooltip="Landfill Trip"
                ).add_to(m)

    @staticmethod
    def _add_legend(m):
        legend_html = """
        <div style="position: fixed; bottom: 50px; right: 50px; width: 200px;
                    background: white; padding: 10px; border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2); z-index: 1000;">
            <h4>Legend</h4>
            <div><span style="display:inline-block;width:12px;height:3px;
                 background:#007bff;margin-right:5px;"></span>Main Route ➔</div>
            <div><span style="display:inline-block;width:12px;height:3px;
                 background:#6c757d;margin-right:5px;border-style:dashed;"></span>Landfill Trip</div>
            <div><span style="display:inline-block;width:12px;height:12px;border:2px solid #28a745;border-radius:50%;margin-right:5px;"></span>SWG Stop</div>
            <div><span style="display:inline-block;width:12px;height:12px;border:2px solid #dc3545;border-radius:50%;margin-right:5px;"></span>DRT Stop</div>
            <div><span style="display:inline-block;width:12px;height:12px;border:2px solid #007bff;border-radius:50%;margin-right:5px;"></span>Hauling Center (H)</div>
            <div><span style="display:inline-block;width:12px;height:12px;border:2px solid #6c757d;border-radius:50%;margin-right:5px;"></span>Landfill (L)</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

# === Helpers ===
def haversine_distance(coord1, coord2):
    R = 6371
    dlat = math.radians(coord2[0] - coord1[0])
    dlon = math.radians(coord2[1] - coord1[1])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(coord1[0])) * math.cos(math.radians(coord2[0])) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def calculate_distance_and_time_matrix(locations):
    num = len(locations)
    dist_matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if i != j:
                dist_matrix[i][j] = haversine_distance(locations[i], locations[j])
    return dist_matrix, dist_matrix / 40  # assume 40 km/h speed

def TSP_solver(customers, cij):
    if not customers:
        return []
    
    try:
        mdl = Model()
        print("Length of Customers:", len(customers))

        n = len(customers)
        customers_idx = list(range(n))  # 0,1,2,...

        # Variables
        x = mdl.addVars(customers_idx, customers_idx, vtype=GRB.BINARY)

        # Auxiliary variables for subtour elimination (MTZ)
        u = mdl.addVars(customers_idx, vtype=GRB.CONTINUOUS, lb=0, ub=n-1)

        # Objective
        mdl.setObjective(quicksum(cij[customers[i], customers[j]] * x[i, j] 
                                  for i in customers_idx for j in customers_idx if i != j), GRB.MINIMIZE)

        # Constraints
        mdl.addConstrs(quicksum(x[i, j] for j in customers_idx if i != j) == 1 for i in customers_idx)
        mdl.addConstrs(quicksum(x[j, i] for j in customers_idx if i != j) == 1 for i in customers_idx)

        # Subtour elimination
        for i in customers_idx[1:]:
            for j in customers_idx[1:]:
                if i != j:
                    mdl.addConstr(u[i] - u[j] + (n-1)*x[i,j] <= n-2)

        mdl.Params.TimeLimit = 90
        mdl.Params.OutputFlag = 0
        mdl.optimize()

        if mdl.Status != GRB.OPTIMAL and mdl.Status != GRB.TIME_LIMIT:
            raise ValueError("Gurobi failed to solve TSP.")

        # --- Build tour without return ---
        tour = []
        current = 0
        visited = set()

        while len(visited) < n:
            visited.add(current)
            for j in customers_idx:
                if current != j and x[current, j].X > 0.5:
                    tour.append(customers[current])
                    current = j
                    break

        # tour.append(customers[current])  # final stop
        return tour

    except Exception as e:
        print("Error:", str(e))
        print("Length of Customers in Fail Cases:", len(customers))
        print("[Fallback] TSP failed, using simple order.")
        return customers


def construct_sequence(arcs, customers):
    if not arcs:
        return customers

    # start from first customer
    sequence = [arcs[0][0], arcs[0][1]]
    arcs = arcs[1:]
    while arcs:
        last_node = sequence[-1]
        found = False
        for i, (start, end) in enumerate(arcs):
            if start == last_node:
                sequence.append(end)
                arcs.pop(i)
                found = True
                break
        if not found:
            # if not found, add any missing customer
            for cust in customers:
                if cust not in sequence:
                    sequence.append(cust)
            break
    return sequence

def create_route_stops(phase: int, sequence, locations, stops_objects) -> List[Stop]:
    newSequence = []
    for i in sequence:
        if i == 0 or i == 1:
            continue  # Skip Haul and LF1
        stop_index = i - 2  # because locations[2] is Stop 1
        if 0 <= stop_index < len(stops_objects):
            newSequence.append(stops_objects[stop_index])
    return newSequence

def save_result_csv(route_id, stops, time_matrix, optimal_route, manual_route, location_id_for_name):
    # Helper to translate index into name
    def translate(idx):
        if idx == 0:
            return "Haul"
        elif idx == 1:
            return "LF1"
        else:
            return f"Stop {idx-1}"

    # Helper to translate index into type
    def translate_type(idx, stops):
        if idx == 0:
            return "Haul"
        elif idx == 1:
            return "LF1"
        else:
            stop = next((s for s in stops if int(s.id) == idx), None)
            if stop:
                return "SWG" if stop.operation_type == "SWG" else "DRT"
            else:
                return ""

    # Build readable routes
    def build_route(route, stops):
        names = [translate(idx) for idx in route]
        types = [translate_type(idx, stops) for idx in route]
        # print(names, types)
        return [names, types]

    # Only fill Manual data for driving time and distance
    _, _ = build_route(optimal_route, stops)  # we still need these
    print(build_route(optimal_route, stops))
    man_time, man_distance = 0, 0
    for i in range(len(manual_route) - 1):
        man_time += time_matrix[manual_route[i]][manual_route[i+1]]
        man_distance += time_matrix[manual_route[i]][manual_route[i+1]] * 40
    
    opt_time, opt_distance = 0, 0    
    for i in range(len(optimal_route) - 1):
        opt_time += time_matrix[optimal_route[i]][optimal_route[i+1]]
        opt_distance += time_matrix[optimal_route[i]][optimal_route[i+1]] * 40

    # Percentage calculations
    total_stops = len(stops)
    num_drt = sum(1 for s in stops if s.operation_type == "DRT")
    num_swg = sum(1 for s in stops if s.operation_type == "SWG")
    perc_drt = round((num_drt / total_stops) * 100, 2) if total_stops else 0
    perc_swg = round((num_swg / total_stops) * 100, 2) if total_stops else 0

    result = pd.DataFrame([{
        "Route_ID": route_id,
        "Driving Time (min) Optimal": opt_time,  # blank
        "Driving Distance (mile) Optimal": opt_distance,
        "Driving Time (min.) Manual": man_time,
        "Driving Distance (mile) Manual": man_distance,
        "Percentage of DRT": perc_drt,
        "Percentage of Swing": perc_swg,
        "Number of Stops": total_stops,
        "Route Optimal": build_route(optimal_route, stops),
        "Route Manual": build_route(manual_route, stops)
    }])
    location_id = location_id_for_name.replace("/", "-")
    save_path = f'services/route_optimization_output/IND_results/IND_results{location_id}.csv'
    result.to_csv(save_path, index=False)
    result.to_csv('IND_results.csv', index=False)
    print(f"[Save] IND_results saved: {save_path}")
    return {
        "Route_ID": route_id,
        "Driving Time (min) Optimal": opt_time,
        "Driving Distance (mile) Optimal": opt_distance,
        "Driving Time (min.) Manual": man_time,
        "Driving Distance (mile) Manual": man_distance,
        "Percentage of DRT": perc_drt,
        "Percentage of Swing": perc_swg,
        "Number of Stops": total_stops,
        "Route Optimal": build_route(optimal_route, stops),
        "Route Manual": build_route(manual_route, stops)
    }

def save_sequence_csv(optimal_route, manual_route, time_matrix, stops, location_id_for_name):
    def translate(idx):
        if idx == 0:
            return "Haul"
        elif idx == 1:
            return "LF1"
        else:
            return f"Stop {idx-1}"

    stop_service_time = {int(s.id): 5 for s in stops}  # Assuming 5 min service time for all stops

    records = []
    for route, label in [(optimal_route, "Optimal"), (manual_route, "Manual")]:
        for i in range(len(route)-1):
            from_idx = route[i]
            to_idx = route[i+1]
            service_time = stop_service_time.get(from_idx, 0) if from_idx > 1 else 0
            record = {
                "Route_ID": location_id_for_name,
                "Route_Type": label,
                "Segment": f"{translate(from_idx)} -> {translate(to_idx)}",
                "Time (min)": round(time_matrix[from_idx][to_idx], 2),
                "Distance (km)": round(time_matrix[from_idx][to_idx] * 40 * 1.60934, 2),  # miles -> km
                "Service Time": service_time,
                "PERM_NOTES": "",
                "NOTE": ""
            }
            records.append(record)
    result = pd.DataFrame(records)
    result.to_csv(f'sequence_row.csv', index=False)
    location_id = location_id_for_name.replace("/", "-")
    result.to_csv(f'services/route_optimization_output/Sequence/sequence_row{location_id}.csv', index=False)
    print(f"[Save] Sequence saved: services/route_optimization_output/Sequence/sequence_row{location_id}.csv")
    
def save_maps(route_id, stops, locations, optimal_route, manual_route, landfill, hauling_loc, original_route_info, route_info):
    optimal_seq = create_route_stops(1, optimal_route, locations, stops)
    manual_seq = create_route_stops(1, manual_route, locations, stops)
    print("asdfasdfasdf", route_id)
    RouteVisualizer.create_map(
        stops=stops,
        sequence=optimal_seq,
        route_info=route_info,
        landfill_locs=[landfill],
        hauling_loc=hauling_loc
    ).save(f"maps/optimal_map{route_id}.html")

    RouteVisualizer.create_map(
        stops=stops,
        sequence=manual_seq,
        route_info=original_route_info,
        landfill_locs=[landfill],
        hauling_loc=hauling_loc
    ).save(f"maps/manual_map{route_id}.html")

def build_manual_route_with_real_volume(stops, route_info, truck_volume_capacity=4):
    route = [0]  # Start at Haul
    current_truck_volume = 0.0  # Volume occupied by full containers
    num_stops = len(stops)

    for idx, stop in enumerate(stops):
        stop_idx = int(stop.id)
        is_swg = route_info.get(stop_idx - 1, 0) == 1
        container_volume = int(stop.container_size) / 10

        route.append(stop_idx)  # Visit stop

        if is_swg:
            current_truck_volume += container_volume
            route.append(1)  # Always go to landfill after SWG

            # Look ahead: check if next stop is SWG or DRT
            if idx + 1 < num_stops:
                next_stop = stops[idx + 1]
                is_next_swg = route_info.get(int(next_stop.id) - 1, 0) == 1
                if not is_next_swg:
                    # Next is DRT ➔ Must go back to Haul
                    route.append(0)
            else:
                # Last stop ➔ Just go back to Haul
                route.append(0)

            current_truck_volume = 0.0  # Truck is emptied after landfill

        else:  # DRT case
            current_truck_volume += container_volume
            route.append(1)  # Go to landfill after picking up DRT
            route.append(stop_idx)  # Go back to current stop (reloading)
            current_truck_volume = 0.0  # Reset after dumping

    # End at Haul if not already
    if route[-1] != 0:
        route.append(0)

    return route

def rotate_tour_to_start(tour, start_node=0):
    if start_node not in tour:
        raise ValueError(f"Start node {start_node} not found in tour.")
    idx = tour.index(start_node)
    rotated = tour[idx:] + tour[:idx]
    return rotated[1:]

# === Main Async Function ===
async def generate_route_map(location_id: str):
    df_routes = pd.read_csv('uploaded_files/transformed_data_snowflk.csv')
    df_routes['Route #'] = df_routes['Route #'].astype(str)
    route_id = location_id.rsplit('-', 1)[0]

    df_route = df_routes[df_routes['Route #'] == route_id]
    return_df =df_route.copy().iloc[0]
    if df_route.empty:
        raise ValueError(f"No data for {route_id}")
    sorted_df = df_route.sort_values(by=['Route #', 'SEQUENCE'])
    service_coords = [(row['Latitude'], row['Longitude']) for _, row in sorted_df.iterrows()]
    service_time = sum([row['SERVICE_WINDOW_TIME'] for _, row in sorted_df.iterrows()])

    midpoint = (df_route.iloc[0]['HL_Lat'], df_route.iloc[0]['HL_Longt'])
    landfill = (df_route.iloc[0]['DF_Lat'], df_route.iloc[0]['DF_Longt'])
    locations = [midpoint, landfill] + service_coords

    stops = []
    for idx, (lat, lon) in enumerate(service_coords):
        row = sorted_df.iloc[idx]
        stops.append(Stop(
            id=str(idx+2),
            latitude=lat,
            longitude=lon,
            container_size=str(row['CURRENT_CONTAINER_SIZE']),
            name=f"Stop {idx+1}",
            can_swing=(row['SERVICE_TYPE_CD'] == 'SWG'),
            current_container=str(row['CURRENT_CONTAINER_SIZE']),
            operation_type=('SWG' if row['SERVICE_TYPE_CD'] == 'SWG' else 'DRT')
        ))
    weight_list = [int(stop.container_size) for stop in stops]
    print(weight_list)
    route_info = {int(stop.id)-1: 1 if stop.operation_type == 'SWG' else 0 for stop in stops}
    dist_matrix, time_matrix = calculate_distance_and_time_matrix(locations)
    swg_stops = [idx+2 for idx, stop in enumerate(stops) if stop.operation_type == 'SWG']
    drt_stops = [idx+2 for idx, stop in enumerate(stops) if stop.operation_type == 'DRT']
    original_route_info = {int(stop.id)-1: 1 if stop.operation_type == 'SWG' else 0 for stop in stops}
    # === Global TSP Part (M + N stops) ===
    all_stops = list(range(2, len(stops)+2))  # IDs starting from 2 (Haul=0, Landfill=1)
    all_stops.append(0)
    # Solve global TSP including all SWG and DRT
    global_sequence = TSP_solver(all_stops, time_matrix)
    global_sequence = rotate_tour_to_start(global_sequence)
    print("Optimized Global sequence (all stops):", global_sequence)

    # Now, based on manual M and N
    M = sum(1 for stop in stops if stop.operation_type == 'SWG')  # Number of SWG in manual
    N = sum(1 for stop in stops if stop.operation_type == 'DRT')  # Number of DRT in manual

    # After TSP, assign operation types
    for idx, stop_id in enumerate(global_sequence):
        real_idx = stop_id - 2  # Because stop IDs are offset by 2
        if idx < M:
            stops[real_idx].operation_type = 'SWG'
        else:
            stops[real_idx].operation_type = 'DRT'

    # Update route_info after reassigning operation_type
    route_info = {int(stop.id)-1: 1 if stop.operation_type == 'SWG' else 0 for stop in stops}

    # Build optimized route based on new SWG/DRT assignment and updated rules
    routeOptimizedNew = [0]  # Start at Haul
    current_volume = 0
    truck_volume_capacity = 4

    for idx, stop_id in enumerate(global_sequence):
        stop = stops[stop_id-2]  # careful offset
        container_volume = int(stop.container_size) / 10

        routeOptimizedNew.append(stop_id)

        # Determine next stop if available
        next_stop_id = global_sequence[idx + 1] if idx + 1 < len(global_sequence) else None
        next_stop = stops[next_stop_id-2] if next_stop_id else None

        if stop.operation_type == 'SWG':

            # Add this block to check for different container sizes between SWGs
            if next_stop and next_stop.operation_type == 'SWG':
                if stop.container_size != next_stop.container_size:
                    routeOptimizedNew.append(1)  # Landfill to empty full container
                    routeOptimizedNew.append(0)  # Haul to load new size
                    current_volume = 0
                    continue
                else:
                    routeOptimizedNew.append(1)  # Landfill to empty full container
            if next_stop and next_stop.operation_type == 'DRT':
                routeOptimizedNew.append(1)
                routeOptimizedNew.append(0)
        elif stop.operation_type == 'DRT':
            # After DRT pickup, go landfill
            routeOptimizedNew.append(1)  # Landfill
            # Then come back to same DRT stop to drop container
            routeOptimizedNew.append(stop_id)

    # After all stops, if not at Haul (0), return to Haul
    if routeOptimizedNew[-1] != 0:
        routeOptimizedNew.append(0)

    manual_route = build_manual_route_with_real_volume(stops, original_route_info)
    location_id_for_name = location_id.replace("/", "-")
    hauling_loc_coord = midpoint
    print("asdfasdfasdf", location_id_for_name)
    print(stops)
    print(locations)
    save_maps(location_id_for_name, stops, locations, routeOptimizedNew, manual_route, landfill, hauling_loc_coord, original_route_info, route_info)
    return_value = save_result_csv(location_id, stops, time_matrix, routeOptimizedNew, manual_route, location_id)
    print(return_value)
    print(return_value["Driving Time (min) Optimal"])
    save_sequence_csv(routeOptimizedNew, manual_route, time_matrix, [], location_id)

    return {
        "Route_ID": route_id,
        "Driving Time (min) Optimal": return_value["Driving Time (min) Optimal"] + service_time / 60,
        "Driving Distance (mile) Optimal": return_value["Driving Distance (mile) Optimal"],
        "Driving Time (min) Manual": return_value["Driving Time (min.) Manual"] + service_time / 60,
        "Driving Distance (mile) Manual": return_value[f"Driving Distance (mile) Manual"],
        "Percentage of DRT": return_value["Percentage of DRT"],
        "Percentage of Swing": return_value["Percentage of Swing"],
        "Number of Stops": return_value["Number of Stops"],
        "Route Optimal": return_value["Route Optimal"],
        "Route Manual": return_value["Route Manual"],
        "DATE": return_df["SERVICE_DATE"],
        "HF_DIVISION_NAME": return_df["HF_DIVISION_NAME"],
        "HF_SITE_NAME": return_df["HF_SITE_NAME"],
        "HF_ADDRESS_LINE1": return_df["HF_ADDRESS_LINE1"],
        "HF_ADDRESS_LINE2": return_df["HF_ADDRESS_LINE2"],
        "HF_ADDRESS_CITY": return_df["HF_ADDRESS_CITY"],
        "HF_ADDRESS_STATE": return_df["HF_ADDRESS_STATE"],
        "HF_ADDRESS_POSTAL_CODE": return_df["HF_ADDRESS_POSTAL_CODE"],
        "DF_FACILITY_NAME": return_df["DF_FACILITY_NAME"],
        "DF_ADDRESS_LINE1": return_df["DF_ADDRESS_LINE1"],
        "DF_ADDRESS_LINE2": return_df["DF_ADDRESS_LINE2"],
        "DF_ADDRESS_CITY": return_df["DF_ADDRESS_CITY"],
        "DF_ADDRESS_STATE": return_df["DF_ADDRESS_STATE"],
        "DF_ADDRESS_POSTAL_CODE": return_df["DF_ADDRESS_POSTAL_CODE"],
        "Time Benefit": return_value["Driving Time (min.) Manual"] - return_value["Driving Time (min) Optimal"],
        "Distance Benefit": return_value[f"Driving Distance (mile) Manual"] - return_value["Driving Distance (mile) Optimal"],
        "Benefit": 0 if return_value["Driving Time (min.) Manual"] - return_value["Driving Time (min) Optimal"] > 0 else 1
    }