import numpy as np 


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371  # Radius of earth
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1

    a = (np.sin(dlat/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2))**2

    d = 2*r*np.arcsin(np.sqrt(a))

    return d

lat1, lon1 = 28.7041, 77.1025  # Delhi
lat2, lon2 = 19.0760, 72.8777  # Mumbai

distance = haversine_distance(lat1, lon1, lat2, lon2)
print(f"Haversine Distance: {distance:.2f} km")
