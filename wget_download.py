"""
    Copyright 2025, Samsung R&D Poland
    License: CC-BY-NC-4.0
"""
import logging
import tarfile
import zipfile
from pathlib import Path

import wget


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wget_download(url: str, output_dir_name: str) -> None:
    output_dir = Path(".") / output_dir_name

    if output_dir.is_dir():
        logger.info("Output directory already exists.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading archive file...")
    wget.download(url, out=str(output_dir))

    archive_file_name = url.split("/")[-1]
    archive_file = output_dir / archive_file_name
    logger.info("Extracting from archive file...")
    af = (
        tarfile.open(archive_file)
        if archive_file_name.endswith(".tar.gz")
        else zipfile.ZipFile(archive_file)
    )
    af.extractall(output_dir)
    af.close()
    archive_file.unlink()
