import sys
from pathlib import Path

from halo import Halo

# local modules
sys.path.append(".")
from src.preprocessing.file_utils import *
from src.preprocessing.preprocess_sat import (
    maxprojection_viirs,
    minmax_scale,
    preprocess_landsat,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_viirs,
    quantile_clip,
)
from src.utilities import SatelliteType
from src.visualization.plot_utils import *

ROOT = Path.cwd()

if __name__ == "__main__":
    train_dir = ROOT / "data" / "raw" / "Train"
    tile1_dir = train_dir / "Tile1"
    save_plots_dir = ROOT / "plots"

    save_plots_dir.mkdir(exist_ok=True)


    with Halo(text="plotting...", spinner="circle") as spinner:

        if False:
            sentinel1_data_array_list = load_satellite_dir(
                train_dir, [SatelliteType.S1]
            )
            sentinel1_data_set = create_satellite_dataset_list(
                sentinel1_data_array_list,
                [SatelliteType.S1],
                list_of_preprocess_func_list=[[preprocess_sentinel1]],
            )[0]
            plot_sentinel1_histogram(sentinel1_data_set, image_dir=None)
        if True:
            sentinel2_data_array_list = load_satellite_dir(
                train_dir, [SatelliteType.S2]
            )
            sentinel2_data_set = create_satellite_dataset_list(
                sentinel2_data_array_list,
                [SatelliteType.S2],
                list_of_preprocess_func_list=[[preprocess_sentinel2]],
            )[0]
            plot_sentinel2_histogram(sentinel2_data_set, image_dir=None)
        if False:
            landsat_data_array_list = load_satellite_dir(
                train_dir, [SatelliteType.LANDSAT]
            )
            landsat_data_set = create_satellite_dataset_list(
                landsat_data_array_list,
                [SatelliteType.LANDSAT],
                list_of_preprocess_func_list=[[preprocess_landsat]],
            )[0]
            plot_landsat_histogram(landsat_data_set, save_plots_dir)
        if False:
            gt_data_array_list = load_satellite_dir(train_dir, [SatelliteType.GT])
            gt_data_set = create_satellite_dataset_list(
                gt_data_array_list, [SatelliteType.GT]
            )[0]
            plot_gt_histogram(gt_data_set, image_dir=None)
        if False:
            gt_tile1_data_array = load_satellite(tile1_dir, SatelliteType.GT)
            plot_gt(gt_tile1_data_array, image_dir=None)
        if False:
            viirs_tile1_data_array = load_satellite(tile1_dir, SatelliteType.VIIRS)
            preprocessed_viirs_tile1_data_array = preprocess_viirs(
                viirs_tile1_data_array
            )
            plot_viirs(preprocessed_viirs_tile1_data_array, image_dir=None)

       
        if False:
            landsat_tile1_data_array = load_satellite(tile1_dir, SatelliteType.LANDSAT)
            preprocessed_landsat_tile1_data_array = preprocess_landsat(
                landsat_tile1_data_array
            )
            plot_satellite_by_bands(
                data_array=preprocessed_landsat_tile1_data_array,
                bands_to_plot=[["4", "3", "2"], ["7", "5", "3"], ["7", "6", "4"]],
                image_dir=save_plots_dir,
            )
        spinner.stop_and_persist(symbol="✅", text="Finished plotting")
