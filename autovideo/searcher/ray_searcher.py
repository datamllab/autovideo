import pandas as pd

from ray import tune

from autovideo.utils import build_pipeline, fit_produce, extract_frames, compute_accuracy_with_preds
from .base import BaseSearcher

class RaySearcher(BaseSearcher):
    def __init__(
        self,
        train_dataset,
        train_media_dir,
        valid_dataset,
        valid_media_dir,
    ):
        super().__init__(
            train_dataset=train_dataset,
            train_media_dir=train_media_dir,
            valid_dataset=valid_dataset,
            valid_media_dir=valid_media_dir
        )

    def search(self, search_space, config):
        if config["searching_algorithm"] == "random":
            from ray.tune.suggest.basic_variant import BasicVariantGenerator
            searcher = BasicVariantGenerator() #Random/Grid Searcher
        elif config["searching_algorithm"] == "hyperopt":
            from ray.tune.suggest.hyperopt import HyperOptSearch
            searcher = HyperOptSearch(max_concurrent=2, metric="accuracy") #HyperOpt Searcher
        else:
            raise ValueError("Searching algorithm not supported.")

        self.valid_labels = self.valid_dataset['label']
        self.valid_dataset = self.valid_dataset.drop(['label'], axis=1)
        
        # Extract frames from the video
        video_ext = self.train_dataset.iloc[0, 1].split('.')[-1]
        extract_frames(self.train_media_dir, video_ext)
        extract_frames(self.valid_media_dir, video_ext)
            
        analysis = tune.run(
            self._evaluate,
            config=search_space,
            num_samples=config["num_samples"],
            resources_per_trial={"cpu": 2, "gpu": 1},
            mode='max',
            search_alg=searcher,
            name=config["searching_algorithm"]+"_"+str(config["num_samples"])
        )
        best_config = analysis.get_best_config(metric="accuracy")
        
        return best_config

    def _evaluate(self, config):
        pipeline = build_pipeline(config)

        # Fit and produce
        predictions = fit_produce(
            train_dataset=self.train_dataset,
            train_media_dir=self.train_media_dir,
            test_dataset=self.valid_dataset,
            test_media_dir=self.valid_media_dir,
            target_index=self.target_index,
            pipeline=pipeline
        )

        # Get accuracy
        valid_acc = compute_accuracy_with_preds(predictions['label'], valid_labels)
        tune.report(accuracy=valid_acc)

