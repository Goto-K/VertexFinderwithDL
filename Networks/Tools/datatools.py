

def LoadPairData(cc03_data_path, cc03_original_data_path, cc04_data_path, cc04_original_data_path,
                 bb04_data_path, bb04_original_data_path, bb05_data_path, bb05_original_data_path):
    cc03data = np.load(cc03_data_path)
    cc03oridata = np.load(cc03_original_data_path)
    cc04data = np.load(cc04_data_path)
    cc04oridata = np.load(cc04_original_data_path)
    bb04data = np.load(bb04_data_path)
    bb04oridata = np.load(bb04_original_data_path)
    bb05data = np.load(bb05_data_path)
    bb05oridata = np.load(bb05_original_data_path)
    
    data = np.concatenate([cc03data, cc04data, bb04data, bb05data], 0)
    original_data = np.concatenate([cc03oridata, cc04oridata, bb04oridata, bb05oridata], 0)

    return data, original_data


def GetPairData(data, original_data, Cov=True):
    if Cov:
        variables = data[:, 3:47]
    else:
        variables = np.concatenate([[:, 3:9], data[:, 25:31]], 0)
    
    vertex = np_utils.to_categorical(data[:, 57], 7)

    original_data[:, 48:51] = TOOLS.cartesian2polar(original_data[:, 48:51])
    position = original_data[:, 48]

    print("Variable Shape: " + str(variables.shape))

    return variables, vertex, position
