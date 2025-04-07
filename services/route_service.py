import csv
from typing import List, Dict, Any
from fastapi import HTTPException
import re
import ast

def get_route_comparison(route_number: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads both IND_results.csv (the aggregate route details) and sequence_row.csv (the detailed segment data)
    and retrieves all rows matching the given Route Number.
    
    Returns a dictionary with two keys:
      - "aggregate": list of dictionaries from IND_results.csv.
      - "sequence": list of dictionaries from sequence_row.csv.
    
    :param route_number: The Route Number to filter by.
    :return: A dictionary with keys "aggregate" and "sequence".
    """
    aggregate_data = []
    sequence_data = []
    try:
        # Read the aggregate results from IND_results.csv
        with open('IND_results.csv', mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row.get("Route_ID") == route_number:
                    # print(123123123)
                    route_optimal_data = ast.literal_eval(row.get("Route Optimal", ""))
                    route_manual_data = ast.literal_eval(row.get("Route Manual", ""))
                    
                    # print("route_manual_dataasdfasdfasdf", route_manual_data)
                    print("route_manual_data", route_manual_data)
                    print("route_manual_data", route_manual_data[0][0])
                    print("route_optimal_data", route_optimal_data)
                    print("route_optimal_data", route_optimal_data[0][0])

                    table_manual_data = []
                    stop_manual_data = []
                    operation_manual_data = []
                    index_manual = []

                    past_index_match = 0
                    stop_length = 1
                    for index, node in enumerate(route_manual_data[0]):
                        if index == 0:
                            continue
                        index_match = re.match(r'Stop (\d+)', node)
                        if index_match:
                            index_match_num = index_match.group(1)
                            if index_match_num == past_index_match:
                                stop_length += 1
                                continue
                            else:
                                stop_manual_data.append(f"Stop {index_match_num}")
                                operation_manual_data.append(route_manual_data[1][index])
                                index_manual.append(index)
                                stop_length = 1
                                past_index_match = index_match_num
                        else:
                            stop_length += 1

                    table_manual_data.append(stop_manual_data)
                    table_manual_data.append(operation_manual_data)
                    table_manual_data.append(index_manual)

                    table_optimal_data = []
                    stop_optimal_data = []
                    operation_optimal_data = []
                    index_optimal = []

                    past_index_match = 0
                    stop_length = 1
                    for index, node in enumerate(route_optimal_data[0]):
                        if index == 0:
                            continue
                        index_match = re.match(r'Stop (\d+)', node)
                        if index_match:
                            index_match_num = index_match.group(1)
                            if index_match_num == past_index_match:
                                stop_length += 1
                                continue
                            else:
                                stop_optimal_data.append(f"Stop {index_match_num}")
                                operation_optimal_data.append(route_optimal_data[1][index])
                                index_optimal.append(index)
                                stop_length = 1
                                past_index_match = index_match_num
                        else:
                            stop_length += 1

                    table_optimal_data.append(stop_optimal_data)
                    table_optimal_data.append(operation_optimal_data)
                    table_optimal_data.append(index_optimal)

                    aggregate_data.append({
                        "Route_ID": row.get("Route_ID", ""),
                        "Driving Time (min) Optimal": float(row.get("Driving Time (min) Optimal", 0)),
                        "Driving Distance (mile) Optimal": float(row.get("Driving Distance (mile) Optimal", 0)),
                        "Driving Time (min.) Manual": float(row.get("Driving Time (min.) Manual", 0)),
                        "Driving Distance (mile) Manual": float(row.get("Driving Distance (mile) Manual", 0)),
                        "Percentage of DRT": float(row.get("Percentage of DRT", 0)),
                        "Percentage of Swing": float(row.get("Percentage of Swing", 0)),
                        "Number of Stops": int(row.get("Number of Stops", 0)),
                        "Route Optimal": row.get("Route Optimal", ""),
                        "Route Manual": row.get("Route Manual", ""),
                        "PERM_NOTES": row.get("PERM_NOTES", ""),
                        "Table Manual Data": table_manual_data,
                        "Table Optimal Data": table_optimal_data,
                    })

        # Read the segment details from sequence_row.csv
        with open('sequence_row.csv', mode="r", encoding="utf-8") as seq_file:
            reader = csv.DictReader(seq_file)
            for row in reader:
                # print(123123123)
                if row.get("Route_ID") == route_number:
                    try:
                        time_val = float(row.get("Time (min)", "0"))
                    except Exception:
                        time_val = 0.0
                    try:
                        dist_val = float(row.get("Distance (km)", "0"))
                    except Exception:
                        dist_val = 0.0
                    sequence_data.append({
                        "Route_ID": row.get("Route_ID", ""),
                        "Route_Type": row.get("Route_Type", ""),
                        "Segment": row.get("Segment", ""),
                        "Time (min)": time_val,
                        "Distance (km)": dist_val,
                        "PERM_NOTES": row.get("PERM_NOTES", ""),
                        "Service Time": row.get("Service Time", "")
                    })
                    
        if not aggregate_data and not sequence_data:
            raise ValueError(f"No records found for Route Number '{route_number}'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV files: {str(e)}")
    
    return {
        "aggregate": aggregate_data,
        "sequence": sequence_data
    }
