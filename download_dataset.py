"""
    Copyright 2025, Samsung R&D Poland
    License: CC-BY-NC-4.0
"""
from wget_download import wget_download


if __name__ == "__main__":
    url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    output_dir_name = "data"
    wget_download(url, output_dir_name)
