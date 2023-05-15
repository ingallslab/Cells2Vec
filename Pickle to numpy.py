import numpy as np
from CellProfilerAnalysis.strain.ProcessCellProfilerData import Dict2Class


def make_numpy_array(pickle_to_dict):
    property_dict = {"stepNum": pickle_to_dict["stepNum"], 'lineage': pickle_to_dict['lineage'], 'id': [], 'label': [],
                     'cellType': [], 'divideFlag': [], 'cellAge': [], 'growthRate': [], 'LifeHistory': [],
                     'startVol': [], 'targetVol': [], 'pos': [], 'time': [], 'radius': [], 'length': [], 'dir': [],
                     'ends': [], 'strainRate': [], 'strainRate_rolling': []}

    for key in pickle_to_dict['cellStates'].keys():
        property_dict['id'].append(pickle_to_dict['cellStates'][key].id)
        property_dict['label'].append(pickle_to_dict['cellStates'][key].label)
        property_dict['cellType'].append(pickle_to_dict['cellStates'][key].cellType)
        property_dict['divideFlag'].append(pickle_to_dict['cellStates'][key].divideFlag)
        property_dict['cellAge'].append(pickle_to_dict['cellStates'][key].cellAge)
        property_dict['growthRate'].append(pickle_to_dict['cellStates'][key].growthRate)
        property_dict['LifeHistory'].append(pickle_to_dict['cellStates'][key].LifeHistory)
        property_dict['startVol'].append(pickle_to_dict['cellStates'][key].startVol)
        property_dict['targetVol'].append(pickle_to_dict['cellStates'][key].targetVol)
        property_dict['pos'].append(pickle_to_dict['cellStates'][key].pos)
        property_dict['time'].append(pickle_to_dict['cellStates'][key].time)
        property_dict['radius'].append(pickle_to_dict['cellStates'][key].radius)
        property_dict['length'].append(pickle_to_dict['cellStates'][key].length)
        property_dict['dir'].append(pickle_to_dict['cellStates'][key].dir)
        property_dict['ends'].append(pickle_to_dict['cellStates'][key].ends)
        property_dict['strainRate'].append(pickle_to_dict['cellStates'][key].strainRate)
        property_dict['strainRate_rolling'].append(pickle_to_dict['cellStates'][key].strainRate_rolling)

    for key in property_dict:
        property_dict[key] = np.array(property_dict[key])

    arr = np.array(list(property_dict.items()), dtype=object)

    return arr


if __name__ == '__main__':
    pickle_file = 'step-000089.pickle'

    pickle_to_dict = np.load(pickle_file, allow_pickle=True)
    arr = make_numpy_array(pickle_to_dict)
    print(arr)
