import numpy as np

from projectaria_tools.core import mps


def bisection_timestamp_search(
    timed_data:list[mps.ClosedLoopTrajectoryPose], query_timestamp_ns: int) -> int:
    """
    Binary search helper function, assuming that timed_data is sorted by the field names 'tracking_timestamp'
    Returns index of the element closest to the query timestamp else returns None if not found (out of time range)
    """
    # Deal with border case
    if timed_data and len(timed_data) > 1:
        first_timestamp = timed_data[0].tracking_timestamp.total_seconds() * 1e9
        last_timestamp = timed_data[-1].tracking_timestamp.total_seconds() * 1e9
        if query_timestamp_ns <= first_timestamp:
            return None
        elif query_timestamp_ns >= last_timestamp:
            return None
    
    # If this is safe we perform the Bisection search
    timestamps_ns = [_.tracking_timestamp.total_seconds() * 1e9 for _ in timed_data]
    nearest_idx = np.searchsorted(timestamps_ns, query_timestamp_ns)  # a[i-1] < v <= a[i]
    
    # decide between a[i-1] and a[i]
    if nearest_idx > 0:
        start_minus_1_timestamp = timestamps_ns[nearest_idx - 1]
        start_timestamp = timestamps_ns[nearest_idx]
        if abs(start_minus_1_timestamp - query_timestamp_ns) < abs(start_timestamp - query_timestamp_ns):
            nearest_idx = nearest_idx - 1
            
    return nearest_idx