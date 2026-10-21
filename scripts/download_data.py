"""Download the catalog CSV and images at the revision pinned in config.yaml."""

from perfume_classifier.config import load_config
from perfume_classifier.data import download

if __name__ == "__main__":
    cfg = load_config()
    download(cfg)
    print(f"CSV and images in {cfg.raw_dir} and {cfg.images_dir}")
