# Base searcher class

import os

class BaseSearcher:
    def __init__(
        self,
        train_dataset,
        train_media_dir,
        valid_dataset,
        valid_media_dir,
    ):
        self.train_dataset = train_dataset
        self.train_media_dir = os.path.abspath(train_media_dir)
        self.valid_dataset = valid_dataset
        self.valid_media_dir = os.path.abspath(valid_media_dir)
        self.target_index = 2

    def search(self, search_space, config):
        raise NotImplementedError     
