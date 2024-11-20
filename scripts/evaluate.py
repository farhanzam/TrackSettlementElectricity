import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_plot


def main(options):
    # initialize datamodule
    datamodule = ESDDataModule(
        options.processed_dir,
        options.raw_dir,
        options.batch_size,
        options.seed,
        options.selected_bands,
        options.slice_size
    )

    # prepare data
    datamodule.prepare_data()
    datamodule.setup('test')

    # load model from checkpoint
    model = ESDSegmentation.load_from_checkpoint(options.model_path)
    # set model to eval mode
    model.eval()

    # get a list of all processed tiles
    tile_dirs = list(Path(options.processed_dir).rglob("*.tif"))
    
    # for each tile
    for tile in tile_dirs:
        # run restitch and plot
        restitch_and_plot(
            options,
            datamodule,
            model,
            tile,
            options.accelerator,
            results_dir=options.results_dir
        )


if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))
