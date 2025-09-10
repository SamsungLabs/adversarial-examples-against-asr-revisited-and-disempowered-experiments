"""
Copyright 2025, Samsung R&D Poland
License: CC-BY-NC-4.0
"""

import logging
from pathlib import Path
from run_all_attacks import AudioAdversarialAttacksTester
from heatmap_creator import HeatmapCreator
from wget_download import wget_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_test():
    logger.info("Downloading dataset...")
    dataset_url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    wget_download(dataset_url, output_dir_name="data")

    logger.info("Downloading checkpoint...")
    checkpoint_url = "http://download.tensorflow.org/models/speech_commands_v0.01.zip"
    wget_download(checkpoint_url, output_dir_name="ckpts")

    tester = AudioAdversarialAttacksTester()
    heatmap_creator = HeatmapCreator()

    logger.info("Running tests of audio adversarial attacks as in the DYHT code...")
    dyht_code_out_dir = Path("./DYHT_code")
    wrong_out_results_matrix_perc, wrong_out_results_df = tester.run_all_attacks(
        output_dir=dyht_code_out_dir, eps_limit=16, endianness="big"
    )
    tester.save_results(
        output_dir=dyht_code_out_dir,
        output_results_matrix_perc=wrong_out_results_matrix_perc,
        output_results_df=wrong_out_results_df,
    )
    heatmap_creator.save_all_heatmaps(results_dir=dyht_code_out_dir)

    logger.info(
        "Running tests of audio adversarial attacks as described in the DYHT paper..."
    )
    dyht_paper_out_dir = Path("./DYHT_paper")
    right_out_results_matrix_perc, right_out_results_df = tester.run_all_attacks(
        output_dir=dyht_paper_out_dir, eps_limit=128, endianness="little"
    )
    tester.save_results(
        output_dir=dyht_paper_out_dir,
        output_results_matrix_perc=right_out_results_matrix_perc,
        output_results_df=right_out_results_df,
    )
    heatmap_creator.save_all_heatmaps(results_dir=dyht_paper_out_dir)


if __name__ == "__main__":
    main_test()
