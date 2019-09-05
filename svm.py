class SVM_Helper:
    def __init__(self, np):
        self.np = np
    
    def parse_request(self, data_Set):

    raw_data = json.loads(data_Set)
    dimensions = len(raw_data[0]) - 1
    #    print(dimensions)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    min_z = float("inf")
    max_z = float("-inf")

    if dimensions == 2:
        for data in raw_data:
            min_x = min(float(min_x), float(data[0]))
            max_x = max(float(max_x), float(data[0]))
            min_y = min(float(min_y), float(data[1]))
            max_y = max(float(max_y), float(data[1]))

        x_y = [[float(raw_data[i][0]),
                float(raw_data[i][1])] for i in range(len(raw_data))]

        labels = [raw_data[i][2] for i in range(len(raw_data))]

        unique_labels = set(labels)
        low_count = min([labels.count(label) for label in unique_labels])
        low_cv = min(5, low_count)

        range_vals = {
            'max_x': max_x,
            'min_x': min_x,
            'max_y': max_y,
            'min_y': min_y
        }
        return (self.np.array(x_y), self.np.array(labels), low_cv, range_vals)

    if dimensions == 3:
        for data in raw_data:
            min_x = min(float(min_x), float(data[0]))
            max_x = max(float(max_x), float(data[0]))
            min_y = min(float(min_y), float(data[1]))
            max_y = max(float(max_y), float(data[1]))
            min_z = min(float(min_z), float(data[2]))
            max_z = max(float(max_z), float(data[2]))

        x_y_z = [[
            float(raw_data[i][0]),
            float(raw_data[i][1]),
            float(raw_data[i][2])
        ] for i in range(len(raw_data))]

        labels = [raw_data[i][3] for i in range(len(raw_data))]

        unique_labels = set(labels)
        low_count = min([labels.count(label) for label in unique_labels])
        low_cv = min(5, low_count)

        range_vals = {
            'max_x': max_x,
            'min_x': min_x,
            'max_y': max_y,
            'min_y': min_y,
            'max_z': max_z,
            'min_z': min_z
        }
        return (self.np.array(x_y_z), self.np.array(labels), low_cv, range_vals)