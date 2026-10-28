from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copytree

import kagglehub

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

with TemporaryDirectory(prefix="kaggle_download_") as download_dir:
    download_path = kagglehub.competition_download(
        "classical-music-meets-classical-ml-fall-2026",
        output_dir=download_dir,
    )
    copytree(download_path, DATA_DIR, dirs_exist_ok=True)

print("Competition files saved to:", DATA_DIR)