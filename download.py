import argparse
import os

from openpi.shared import download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="gs://openpi-assets/checkpoints/pi05_base",
        help="Remote checkpoint or asset URL (e.g. gs://...).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local download root. If set, overrides OPENPI_DATA_HOME.",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["OPENPI_DATA_HOME"] = args.cache_dir

    checkpoint_dir = download.maybe_download(args.url, force_download=args.force_download)
    print(checkpoint_dir)


if __name__ == "__main__":
    main()
