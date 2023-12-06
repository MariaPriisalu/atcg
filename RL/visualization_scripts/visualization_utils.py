import numpy as np
def trendline(data):
    trend_avg = np.zeros(len(data))
    trend_avg[0] = data[0]
    for i in range(1, len(data)):
        trend_avg[i] = (trend_avg[i - 1] * (i - 1.0) + data[i]) / i
    return trend_avg

def get_constants():
    # Pedestrian actions
    actions = []
    v = [-1, 0, 1]
    j = 0
    for y in range(3):
        for x in range(3):
            actions.append(np.array([0, v[y], v[x]]))
            j += 1
    actions_names = ['upL', 'up', 'upR', 'left', 'stand', 'right', 'downL', 'down', 'downR']
    # Semantic labels
    labels = ['human hist', 'car hist', 'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
              'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
              'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign',
              'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
              'train', 'motorcycle', 'bicycle', 'license plate']
    labels_indx = [11, 13, 4, 17, 7, 8, 21, 12, 20]
    labels_mini = []
    labels_mini.append("R")
    labels_mini.append("G")
    labels_mini.append("B")
    labels_mini.append('people-t')
    labels_mini.append('cars-t')
    labels_mini.append('people')
    labels_mini.append('cars')
    for label in labels_indx:
        # print labels[label+2]
        labels_mini.append(labels[label + 2])
    labels_mini.append("prior")
    train_nbrs = [6, 7, 8, 9]  # , 0,2,4,5]
    test_nbrs = [3, 0, 2, 4, 5]
    init_names = {-1: "training", 0: "average", 1: "On ped.", 2: "By car", 3: "In front of ped.", 4: "Random",
                  5: "In front of car", 6: "Near ped.", 7: "ped. env.", 8: "On pavement", 9: "On ped.",
                  10: "Near obstacles", 11: "average"}
    return actions, actions_names, labels, labels_mini, train_nbrs, test_nbrs, init_names
