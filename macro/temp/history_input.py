    name_track_labels = "/home/goto/ILC/Deep_Learning/test_track_labels.npy"
    name_tracks = "/home/goto/ILC/Deep_Learning/test_tracks.npy"
    name_track_pairs = "/home/goto/ILC/Deep_Learning/test_track_pairs.npy"

    data_track_labels = np.load(name_track_labels, allow_pickle=True)
    data_tracks = np.load(name_tracks, allow_pickle=True)
    data_track_pairs = np.load(name_track_pairs, allow_pickle=True)

    print("file load !")

    print(data_track_labels.shape)
    print(data_tracks.shape)
    print(data_track_pairs.shape)


    teach_track_labels = []
    teach_tracks = []
    teach_track_pairs = []
    for labels, tracks, pairs in zip(data_track_labels, data_tracks, data_track_pairs):
        for label, pair in zip(labels, pairs):
            pair = np.array(pair).reshape([44, 1])
            teach_track = []
            teach_track_label = []
            if len(label) < 10:
                continue
            for i, (l, track) in enumerate(zip(label, tracks)):
                if i == 10:  break
                teach_track.append(track)
                teach_track_label.append(l)

            teach_track_labels.append(teach_track_label)
            teach_tracks.append(teach_track)
            teach_track_pairs.append(pair)

    teach_track_labels = np.array(teach_track_labels)
    teach_tracks = np.array(teach_tracks)
    teach_track_pairs = np.array(teach_track_pairs)
