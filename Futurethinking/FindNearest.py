import numpy as np

def find_nearest(array, value, length):
    nearest_veh = []

    array1 = np.asarray([array[i] for i in range(len(array))])
    array_dis = np.asarray([array[i][1] for i in range(len(array))])

    if len(array1):
        idx_l = ((array_dis - value) % length).argmin()
        nearest_veh.append(array1[idx_l])
    else:
        nearest_veh.append([-100, -100, -100])

    if len(array1):
        idx_f = ((value - array_dis) % length).argmin()

        nearest_veh.append(array1[idx_f])
    else:
        nearest_veh.append([-100, -100, -100])

    return nearest_veh