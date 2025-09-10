"""
Copyright 2025, Samsung R&D Poland
License: CC-BY-NC-4.0
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import audio_attack


@dataclass
class HeatmapCreator:
    checkpoints_dir: Optional[Path] = None

    def __post_init__(self):
        if self.checkpoints_dir is None:
            self.checkpoints_dir = Path("./ckpts")

        labels_file = self.checkpoints_dir / "conv_actions_labels.txt"
        self.labels = audio_attack.load_labels(str(labels_file))[2:]
        self.existing_labels_num = len(self.labels)

    def save_heatmap(
        self,
        output_results_matrix_perc: np.ndarray,
        output_results_heatmap_file: Path,
        show: bool = False,
        decimals: int = 0,
    ) -> None:
        font = {"weight": "bold", "size": 12}
        matplotlib.rc("font", **font)
        plt.subplots(figsize=(10, 10))
        sns.heatmap(
            output_results_matrix_perc,
            annot=True,
            fmt=f"0.{decimals}f",
            yticklabels=self.labels,
            xticklabels=self.labels,
            cbar=False,
            cmap="OrRd",
            vmin=0,
            vmax=100.0,
        )
        plt.xlabel("Target Label", fontsize=16)
        plt.ylabel("Source Label", fontsize=16)
        plt.savefig(output_results_heatmap_file)
        if show:
            plt.show()
        plt.close()

    def get_no_attacks_results_matrix(self, attacks_results_df_file: Path) -> np.ndarray:
        attacks_results_df = pd.read_csv(attacks_results_df_file)

        no_attacks_results_matrix_perc = np.zeros(
            [self.existing_labels_num, self.existing_labels_num]
        )
        for i, target_label in enumerate(self.labels):
            for j, source_label in enumerate(self.labels):
                if i == j:
                    continue

                interesting_rows = attacks_results_df.loc[
                    (attacks_results_df["SourceLabel"] == source_label)
                    & (attacks_results_df["TargetLabel"] == target_label)
                ]
                no_attacks_results_matrix_perc[j][i] = (
                    100
                    * interesting_rows.loc[
                        interesting_rows["NoAttackAsrLabel"] == target_label
                    ].shape[0]
                    / interesting_rows.shape[0]
                )
        return no_attacks_results_matrix_perc

    def save_all_heatmaps(
        self,
        results_dir: Path,
        output_results_files_name: str = "output_results",
        show: bool = False,
    ) -> None:
        self.attacks_results_matrix_file = results_dir / f"{output_results_files_name}.npy"

        attacks_results_matrix_perc = np.load(self.attacks_results_matrix_file)
        attacks_results_heatmap_file = (
            self.attacks_results_matrix_file.parent / "attacks_results_matrix.png"
        )
        self.save_heatmap(attacks_results_matrix_perc, attacks_results_heatmap_file, show)
        overall_attacks_result_perc = (
            np.sum(attacks_results_matrix_perc)
            / self.existing_labels_num
            / (self.existing_labels_num - 1)
        )

        attacks_results_df_file = results_dir / f"{output_results_files_name}.csv"

        no_attacks_results_matrix_perc = self.get_no_attacks_results_matrix(
            attacks_results_df_file
        )
        no_attacks_results_heatmap_file = (
            self.attacks_results_matrix_file.parent / "no_attacks_results_matrix.png"
        )
        self.save_heatmap(no_attacks_results_matrix_perc, no_attacks_results_heatmap_file, show)
        overall_no_attacks_result_perc = (
            np.sum(no_attacks_results_matrix_perc)
            / self.existing_labels_num
            / (self.existing_labels_num - 1)
        )

        difference_results_matrix_perc = (
            100
            * (attacks_results_matrix_perc - no_attacks_results_matrix_perc)
            / (100 - no_attacks_results_matrix_perc)
        )
        difference_results_heatmap_file = (
            self.attacks_results_matrix_file.parent / "difference_results_matrix.png"
        )
        self.save_heatmap(
            difference_results_matrix_perc, difference_results_heatmap_file, show, decimals=2
        )
        overall_difference_result_perc = (
            overall_attacks_result_perc - overall_no_attacks_result_perc
        )
        with open(results_dir / "overall_results_perc.txt", "w") as f:
            f.write(f"Attack    : {overall_attacks_result_perc}\n")
            f.write(f"No Attack : {overall_no_attacks_result_perc}\n")
            f.write(f"Difference: {overall_difference_result_perc}\n")


if __name__ == "__main__":
    parsed_args = ArgumentParser()
    parsed_args.add_argument("--results_dir", type=Path, required=True)
    parsed_args.add_argument("--checkpoints_dir", type=Optional[Path], default=None)
    parsed_args.add_argument("--output_results_files_name", type=str, default="output_results")
    parsed_args.add_argument("--show", action="store_true", default=False)
    args = parsed_args.parse_args()

    results_viewer = HeatmapCreator(args.checkpoints_dir)
    results_viewer.save_all_heatmaps(args.results_dir, args.output_results_files_name, args.show)
