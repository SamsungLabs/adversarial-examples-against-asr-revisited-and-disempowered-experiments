"""
    Copyright 2025, Samsung R&D Poland
    License: CC-BY-NC-4.0
"""
from dataclasses import dataclass
import logging
import random
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path
from typing_extensions import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

import audio_attack

INPUT_NODE_NAME = "wav_data:0"
OUTPUT_NODE_NAME = "labels_softmax:0"

Endianness = Literal["big", "little"]
OutputExistsAction = Literal["overwrite", "error"]


class DirectoryExistsError(Exception):
    pass


class TargetLabelNotFoundError(Exception):
    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioAdversarialAttacksTester:
    dataset_dir: Optional[Path] = None
    checkpoints_dir: Optional[Path] = None
    test_size: int = 50
    max_iters: int = 500
    seed: int = 42
    output_exists_action: OutputExistsAction = "overwrite"

    def set_input_data_for_labels(self) -> None:
        self.labels_files_map: Dict[str : List[Path]] = {}
        for label in self.interesting_labels:
            input_label_dir = self.dataset_dir / label
            input_label_files = sorted(input_label_dir.glob("*"))
            random.Random(self.seed).shuffle(input_label_files)
            self.labels_files_map[label] = sorted(input_label_files[: self.test_size])

    def __post_init__(self):
        if self.dataset_dir is None:
            self.dataset_dir = Path("./data")
        if self.checkpoints_dir is None:
            self.checkpoints_dir = Path("./ckpts")

        labels_file = self.checkpoints_dir / "conv_actions_labels.txt"
        self.all_labels = audio_attack.load_labels(str(labels_file))
        self.interesting_labels = self.all_labels[2:]

        self.set_input_data_for_labels()

        graph_path = self.checkpoints_dir / "conv_actions_frozen.pb"
        audio_attack.load_graph(str(graph_path))

    def handle_output_dir(self, output_dir: Path) -> Tuple[Path, Path]:
        if output_dir is None:
            output_dir = Path("./output")
        if output_dir.is_dir():
            if self.output_exists_action == "overwrite":
                shutil.rmtree(output_dir)
            elif self.output_exists_action == "error":
                raise DirectoryExistsError(
                    "Output directory already exists and 'error' action specified."
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def find_target_label_idx(self, target_label: str) -> int:
        for idx, label in enumerate(self.all_labels):
            if label == target_label:
                return idx
        raise TargetLabelNotFoundError(
            "Target label not found in labels known to tested model."
        )

    def generate_single_attack_on_original_audio(
        self,
        original_audio: np.ndarray,
        input_file: Path,
        current_output_dir: Path,
        target_label_idx: int,
        eps_limit: int,
        endianness: Endianness,
        tf_session: tf.Session,
        output_node,
    ) -> Tuple[int, int]:
        start_time = time.time()
        attack_output, iter_idx = audio_attack.generate_attack(
            x_orig=original_audio,
            target=target_label_idx,
            limit=0,
            sess=tf_session,
            input_node=INPUT_NODE_NAME,
            output_node=output_node,
            max_iters=self.max_iters,
            eps_limit=eps_limit,
            verbose=False,
            endianness=endianness,
        )
        audio_attack.save_audiofile(attack_output, current_output_dir / input_file.name)
        end_time = time.time()
        return iter_idx, end_time - start_time

    def get_asr_label_when_no_attack(
        self, original_audio: np.ndarray, tf_session: tf.Session, output_node
    ) -> str:
        scores = audio_attack.score(
            sess=tf_session,
            x=original_audio,
            target=None,
            input_tensor=INPUT_NODE_NAME,
            output_tensor=output_node,
        )
        return self.all_labels[np.argmax(scores)]

    def handle_single_attack_result(
        self,
        original_audio: np.ndarray,
        iter_idx: int,
        output_results_df: pd.DataFrame,
        source_label: str,
        target_label: str,
        input_file_name: str,
        attack_duration: float,
        tf_session: tf.Session,
        output_node,
    ) -> None:
        msg = f"File {input_file_name}. {{}}"
        no_attack_asr_label = self.get_asr_label_when_no_attack(
            original_audio,
            tf_session,
            output_node,
        )
        if iter_idx == -1:
            logger.info(msg.format("Attack failed"))
            variable1 = False
            iter_idx = None
            attack_duration = None
        else:
            logger.info(msg.format(f"Attack succeeded in {iter_idx} iterations ({attack_duration:.2f} seconds)"))
            variable1=True,
        values = [
            source_label,
            target_label,
            input_file_name,
            variable1,
            iter_idx,
            attack_duration,
            no_attack_asr_label,
        ]
        output_results_df.loc[len(output_results_df)] = values

    def run_all_attacks(
        self,
        output_dir: Path = None,
        eps_limit: int = 16,
        endianness: Endianness = "big",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        output_dir = self.handle_output_dir(output_dir)

        logger.info(f"""Running all attacks:
- dataset_dir={self.dataset_dir},
- checkpoints_dir={self.checkpoints_dir},
- output_dir={output_dir},
- test_size={self.test_size},
- max_iters={self.max_iters},
- seed={self.seed},
- eps_limit={eps_limit},
- endianness={endianness},
- output_exists_action={self.output_exists_action}.
""")

        existing_labels_num = len(self.labels_files_map)
        output_results_matrix_perc = np.zeros(
            [existing_labels_num, existing_labels_num]
        )
        output_results_df = pd.DataFrame(
            columns=[
                "SourceLabel",
                "TargetLabel",
                "SourceFileName",
                "Success",
                "Iterations",
                "Duration",
                "NoAttackAsrLabel",
            ]
        )

        with tf.Session() as tf_session:
            output_node = tf_session.graph.get_tensor_by_name(OUTPUT_NODE_NAME)

            for i, target_label in enumerate(self.labels_files_map):
                target_label_idx = self.find_target_label_idx(target_label)

                for j, source_label in enumerate(self.labels_files_map):
                    if target_label == source_label:
                        continue

                    logger.info(f"Running attack: {source_label} --> {target_label}")

                    current_output_dir = output_dir / target_label / source_label
                    current_output_dir.mkdir(parents=True, exist_ok=True)

                    succeeded_cnt = 0
                    for input_file in self.labels_files_map[source_label]:
                        original_audio = audio_attack.load_audiofile(input_file)

                        iter_idx, attack_duration = (
                            self.generate_single_attack_on_original_audio(
                                original_audio,
                                input_file,
                                current_output_dir,
                                target_label_idx,
                                eps_limit,
                                endianness,
                                tf_session,
                                output_node,
                            )
                        )
                        succeeded_cnt += int(iter_idx != -1)
                        self.handle_single_attack_result(
                            original_audio,
                            iter_idx,
                            output_results_df,
                            source_label,
                            target_label,
                            input_file.stem,
                            attack_duration,
                            tf_session,
                            output_node,
                        )

                    logger.info(f"Succeeded: {succeeded_cnt}/{self.test_size}")
                    output_results_matrix_perc[j, i] = (
                        100 * succeeded_cnt / self.test_size
                    )

        return output_results_matrix_perc, output_results_df

    def save_results(
        self,
        output_dir: Path,
        output_results_matrix_perc: np.ndarray,
        output_results_df: pd.DataFrame,
        output_results_files_name: str = "output_results",
    ) -> None:
        logger.info(
            f"Saving results, output_results_files_name={output_results_files_name}"
        )
        np.save(
            output_dir / f"{output_results_files_name}.npy", output_results_matrix_perc
        )
        output_results_df.to_csv(
            output_dir / f"{output_results_files_name}.csv", index=False
        )


if __name__ == "__main__":
    parsed_args = ArgumentParser()
    parsed_args.add_argument("--dataset_dir", type=Optional[Path], default=None)
    parsed_args.add_argument("--checkpoints_dir", type=Optional[Path], default=None)
    parsed_args.add_argument("--output_dir", type=Optional[Path], default=None)
    parsed_args.add_argument(
        "--output_results_files_name", type=str, default="output_results"
    )
    parsed_args.add_argument("--test_size", type=int, default=50)
    parsed_args.add_argument("--max_iters", type=int, default=500)
    parsed_args.add_argument("--seed", type=int, default=42)
    parsed_args.add_argument("--eps_limit", type=int, default=16)
    parsed_args.add_argument(
        "--endianness", type=str, choices=["big", "little"], default="big"
    )
    parsed_args.add_argument(
        "--output_exists_action",
        type=str,
        choices=["overwrite", "error"],
        default="overwrite",
    )
    args = parsed_args.parse_args()

    tester = AudioAdversarialAttacksTester(
        args.dataset_dir,
        args.checkpoints_dir,
        args.test_size,
        args.max_iters,
        args.seed,
        args.output_exists_action,
    )
    output_results_matrix_perc, output_results_df = tester.run_all_attacks(
        args.output_dir,
        args.eps_limit,
        args.endianness,
    )
    tester.save_results(
        args.output_dir,
        output_results_matrix_perc,
        output_results_df,
        args.output_results_files_name,
    )
