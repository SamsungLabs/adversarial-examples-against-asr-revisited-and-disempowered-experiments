"""
    Copyright 2025, Samsung R&D Poland
    License: CC-BY-NC-4.0
"""
from wget_download import wget_download


if __name__ == "__main__":
    url = "http://download.tensorflow.org/models/speech_commands_v0.01.zip"
    output_dir_name = "ckpts"
    wget_download(url, output_dir_name)
