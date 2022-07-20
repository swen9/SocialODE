import pandas
import glob
import numpy as np
import torch
import torch.nn.functional as functional
import os, argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils



def read_all_recordings_from_csv(base_path="../data/"):
    """
    This methods reads the tracks and meta information for all recordings given the path of the inD dataset.
    :param base_path: Directory containing all csv files of the inD dataset
    :return: a tuple of tracks, static track info and recording meta info
    """
    tracks_files = sorted(glob.glob(base_path + "*_tracks.csv"))
    static_tracks_files = sorted(glob.glob(base_path + "*_tracksMeta.csv"))
    recording_meta_files = sorted(glob.glob(base_path + "*_recordingMeta.csv"))

    all_tracks = []
    all_static_info = []
    all_meta_info = []
    for track_file, static_tracks_file, recording_meta_file in zip(tracks_files,
                                                                   static_tracks_files,
                                                                   recording_meta_files):
        tracks, static_info, meta_info = read_from_csv(track_file, static_tracks_file, recording_meta_file)
        all_tracks.append(tracks)
        all_static_info.append(static_info)
        all_meta_info.append(meta_info)

    return all_tracks, all_static_info, all_meta_info


def read_from_csv(track_file, static_tracks_file, recordings_meta_file):
    """
    This method reads tracks including meta data for a single recording from csv files.

    :param track_file: The input path for the tracks csv file.
    :param static_tracks_file: The input path for the static tracks csv file.
    :param recordings_meta_file: The input path for the recording meta csv file.
    :return: tracks, static track info and recording info
    """
    static_info = read_static_info(static_tracks_file)
    meta_info = read_meta_info(recordings_meta_file)
    tracks = read_tracks(track_file, meta_info)
    return tracks, static_info, meta_info


def read_tracks(track_file, meta_info):
    # Read the csv file to a pandas dataframe
    df = pandas.read_csv(track_file)

    # To extract every track, group the rows by the track id
    raw_tracks = df.groupby(["trackId"], sort=False)
    ortho_px_to_meter = meta_info["orthoPxToMeter"]
    tracks = []
    for track_id, track_rows in raw_tracks:
        track = track_rows.to_dict(orient="list")

        # Convert scalars to single value and lists to numpy arrays
        for key, value in track.items():
            if key in ["trackId", "recordingId"]:
                track[key] = value[0]
            else:
                track[key] = np.array(value)

        track["center"] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
        #track["bbox"] = calculate_rotated_bboxes(track["xCenter"], track["yCenter"],
        #                                         track["length"], track["width"],
        #                                         np.deg2rad(track["heading"]))

        # Create special version of some values needed for visualization
        track["xCenterVis"] = track["xCenter"] / ortho_px_to_meter
        track["yCenterVis"] = -track["yCenter"] / ortho_px_to_meter
        track["centerVis"] = np.stack([track["xCenter"], -track["yCenter"]], axis=-1) / ortho_px_to_meter
        track["widthVis"] = track["width"] / ortho_px_to_meter
        track["lengthVis"] = track["length"] / ortho_px_to_meter
        track["headingVis"] = track["heading"] * -1
        track["headingVis"][track["headingVis"] < 0] += 360
        #track["bboxVis"] = calculate_rotated_bboxes(track["xCenterVis"], track["yCenterVis"],
        #                                            track["lengthVis"], track["widthVis"],
        #                                            np.deg2rad(track["headingVis"]))

        tracks.append(track)
    return tracks


def read_static_info(static_tracks_file):
    """
    This method reads the static info file from highD data.

    :param static_tracks_file: the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    return pandas.read_csv(static_tracks_file).to_dict(orient="records")


def read_meta_info(recordings_meta_file):
    """
    This method reads the recording info file from ind data.

    :param recordings_meta_file: the path for the recording meta csv file.
    :return: the meta dictionary
    """
    return pandas.read_csv(recordings_meta_file).to_dict(orient="records")[0]

class indDataset_train(Dataset):
    # inD Dataset
    def __init__(self, path):
        super(indDataset_train, self).__init__()
        self.all_tracks, self.all_static, self.all_meta = read_all_recordings_from_csv(path)
        self.train_tracks = self.all_tracks[0:24]
        self.num_tracks = 0
        self.track_list = []
        self.x_min = 100000
        self.x_max = -100000
        self.y_min = 100000
        self.y_max = -100000
        for self.track_set_id, self.track_set in enumerate(self.train_tracks):
            self.num_tracks = self.num_tracks + len(self.track_set)
            for track_id, track in enumerate(self.track_set):
                if track['trackLifetime'][-1] < 200 :
                    self.num_tracks = self.num_tracks - 1
                    continue
                #print (track['xCenter'][0:200:10])
                if track['xCenter'][0:200:10].min() < self.x_min :
                    self.x_min = track['xCenter'][0:200:10].min()
                if track['yCenter'][0:200:10].min() < self.y_min :
                    self.y_min = track['yCenter'][0:200:10].min()                
                if track['xCenter'][0:200:10].max() > self.x_max :
                    self.x_max = track['xCenter'][0:200:10].max()
                if track['yCenter'][0:200:10].max() > self.y_max :
                    self.y_max = track['yCenter'][0:200:10].max()


                self.track_list.append(torch.FloatTensor(np.stack([track['xCenter'][0:200:10],track['yCenter'][0:200:10],track['xVelocity'][0:200:10],track['yVelocity'][0:200:10]], axis=-1)))
                

    def __len__(self):
        return int(self.num_tracks)

    def __getitem__(self, idx):
        return self.track_list[idx], self.x_min, self.x_max, self.y_min, self.y_max


class indDataset_test(Dataset):
    # inD Dataset
    def __init__(self, path):
        super(indDataset_test, self).__init__()
        self.all_tracks, self.all_static, self.all_meta = read_all_recordings_from_csv(path)
        self.test_tracks = self.all_tracks[25:30]
        self.num_tracks = 0
        self.track_list = []
        for self.track_set_id, self.track_set in enumerate(self.test_tracks):
            self.num_tracks = self.num_tracks + len(self.track_set)
            for track_id, track in enumerate(self.track_set):
                if track['trackLifetime'][-1] < 200 :
                    self.num_tracks = self.num_tracks - 1
                    continue
                #print (track['xCenter'][0:200:10])
                if track['xCenter'][0:200:10].min() < self.x_min :
                    self.x_min = track['xCenter'][0:200:10].min()
                if track['yCenter'][0:200:10].min() < self.y_min :
                    self.y_min = track['yCenter'][0:200:10].min()                
                if track['xCenter'][0:200:10].max() > self.x_max :
                    self.x_max = track['xCenter'][0:200:10].max()
                if track['yCenter'][0:200:10].max() > self.y_max :
                    self.y_max = track['yCenter'][0:200:10].max()
                self.track_list.append(torch.FloatTensor(np.stack([track['xCenter'][0:200:10],track['yCenter'][0:200:10],track['xVelocity'][0:200:10],track['yVelocity'][0:200:10]], axis=-1)))
                

    def __len__(self):
        return int(self.num_tracks)

    def __getitem__(self, idx):
        return self.track_list[idx], self.x_min, self.x_max, self.y_min, self.y_max
