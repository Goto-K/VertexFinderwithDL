import numpy as np


if __name__ == "__main__":
    # The track list is not sorted
    variable_name = "/home/goto/ILC/Deep_Learning/data/vfdnn/vfdnn_03.npy"

    data = np.load(variable_name)
    print("File load successfully")

    event_nums = list(range(10000))
    print("Making event number list")

    tracks = []
    track_labels = []
    track_pairs = []

    for event_num in event_nums:
        event_data = [datum for datum in data if datum[0]==event_num]

        track_num = (1 + np.sqrt(1 + 8*len(event_data)))/2 # 1,2,3,4,5,...
        print("The number of tracks in event {} is {}".format(event_num, track_num))
        vertex_mat = [[0 for j in range(int(track_num))] for i in range(int(track_num))] # Max number of tracks
        
        track = []
        track_label = []
        track_pair = []

        for event_datum in event_data:
            connect = 1 if event_datum[57] != 0 else 0
            vertex_mat[int(event_datum[1])][int(event_datum[2])] = connect
            vertex_mat[int(event_datum[2])][int(event_datum[1])] = connect
            if int(event_datum[1]) == int(track_num)-1:
                #track.append([event_num, event_datum[2], event_datum[25:47]])
                track.append(event_datum[25:47])
                if int(event_datum[2]) == int(track_num)-2:
                    track.append(event_datum[3:25])

        for t in range(int(track_num)):
            tlist = vertex_mat[t]
            index = [i for i, x in enumerate(tlist) if x == 1]
            if not index:
                continue
            else:
                vertex_mat[t][t] = 1
                track_label.append(vertex_mat[t])
                track_pair.append([track[t], track[index[0]]])

        tracks.append(track)
        track_labels.append(track_label)
        track_pairs.append(track_pair)

    tracks = np.array(tracks)
    track_labels = np.array(track_labels)
    track_pairs = np.array(track_pairs)

    np.save("test_tracks.npy", tracks, fix_imports=True)
    np.save("test_track_labels.npy", track_labels, fix_imports=True)
    np.save("test_track_pairs.npy", track_pairs, fix_imports=True)
