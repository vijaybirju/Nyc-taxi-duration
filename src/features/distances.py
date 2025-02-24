import numpy as np 


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distances between two points given their latitude and
    longitude coordinates
    """
    R = 6371  # Radius of earth
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1

    a = (np.sin(dlat/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2))**2

    d = 2*R*np.arcsin(np.sqrt(a))

    return d


def euclidean_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate euclidean distances between two points given their latitude and
    longitude coordinates
    """
    r = 111  # Approximate km per degree
    dlon = (lon2 - lon1)*r*np.cos(np.radians((lat1 + lat2) / 2)) 
    dlat = (lat2 - lat1)*r

    a = dlon**2 + dlat**2

    d = np.sqrt(a)

    return d

def manhattan_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate manhattan distances between two points given their latitude and
    longitude coordinates
    """
    r = 111  # Approximate km per degree
    dlon = (lon2 - lon1)*r*np.cos(np.radians((lat1 + lat2) / 2)) 
    dlat = (lat2 - lat1)*r

    d = np.abs(dlon) + np.abs(dlat)

    return d



