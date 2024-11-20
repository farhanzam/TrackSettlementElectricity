import sys
from pathlib import Path

from halo import Halo

# local modules
sys.path.append(".")
from src.preprocessing.file_utils import *
from src.preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_landsat,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_viirs,
)
from src.utilities import SatelliteType
from src.visualization.plot_utils import *

ROOT = Path.cwd()
feedback = True

if __name__ == "__main__":
    train_dir = ROOT / "data" / "raw" / "Train"
    tile1_dir = train_dir / "Tile1"
    save_plots_dir = ROOT / "plots"

    save_plots_dir.mkdir(exist_ok=True)

    with Halo(text="Plotting VIIRS data", spinner="circle") as spinner:
        viirs_data_array_list = load_satellite_dir(train_dir, [SatelliteType.VIIRS])
        viirs_data_set = create_satellite_dataset_list(
            viirs_data_array_list,
            [SatelliteType.VIIRS],
            list_of_preprocess_func_list=[[preprocess_viirs]],
        )[0]
        viirs_tile1_data_array = viirs_data_set["Tile1"]
        viirs_tile1_max_projection = maxprojection_viirs(viirs_tile1_data_array)
        plot_viirs_histogram(viirs_data_set, image_dir=save_plots_dir)
        plot_max_projection_viirs(viirs_tile1_max_projection, image_dir=save_plots_dir)
        plot_viirs(viirs_tile1_data_array, image_dir=save_plots_dir)
        spinner.stop_and_persist(symbol="✅", text="Finished plotting VIIRS data")

    with Halo(text="Plotting Sentinel 1 data", spinner="circle") as spinner:
        sentinel1_data_array_list = load_satellite_dir(train_dir, [SatelliteType.S1])
        sentinel1_data_set = create_satellite_dataset_list(
            sentinel1_data_array_list,
            [SatelliteType.S1],
            list_of_preprocess_func_list=[[preprocess_sentinel1]],
        )[0]
        sentinel1_tile1_data_array = sentinel1_data_set["Tile1"]
        plot_sentinel1_histogram(sentinel1_data_set, image_dir=save_plots_dir)
        create_rgb_composite_s1(sentinel1_tile1_data_array, image_dir=save_plots_dir)
        spinner.stop_and_persist(symbol="✅", text="Finished plotting Sentinel 1 data")

    with Halo(text="Plotting Sentinel 2 data", spinner="circle") as spinner:
        sentinel2_data_array_list = load_satellite_dir(train_dir, [SatelliteType.S2])
        sentinel2_data_set = create_satellite_dataset_list(
            sentinel2_data_array_list,
            [SatelliteType.S2],
            list_of_preprocess_func_list=[[preprocess_sentinel2]],
        )[0]
        sentinel2_tile1_data_array = sentinel2_data_set["Tile1"]
        plot_sentinel2_histogram(sentinel2_data_set, image_dir=save_plots_dir)
        plot_satellite_by_bands(
            data_array=sentinel2_tile1_data_array,
            bands_to_plot=[
                ["04", "03", "02"],
                ["12", "8A", "04"],
                ["08", "04", "03"],
            ],
            image_dir=save_plots_dir,
        )
        spinner.stop_and_persist(symbol="✅", text="Finished plotting Sentinel 2 data")

    with Halo(text="Plotting Landsat 8 data", spinner="circle") as spinner:
        sentinel2_data_array_list = load_satellite_dir(
            train_dir, [SatelliteType.LANDSAT]
        )
        landsat_data_set = create_satellite_dataset_list(
            sentinel2_data_array_list,
            [SatelliteType.LANDSAT],
            list_of_preprocess_func_list=[[preprocess_landsat]],
        )[0]
        landsat_tile1_data_array = landsat_data_set["Tile1"]
        plot_landsat_histogram(landsat_data_set, save_plots_dir)
        plot_satellite_by_bands(
            data_array=landsat_tile1_data_array,
            bands_to_plot=[["4", "3", "2"], ["7", "5", "3"], ["7", "6", "4"]],
            image_dir=save_plots_dir,
        )
        spinner.stop_and_persist(symbol="✅", text="Finished plotting Landsat 8 data")

    with Halo(text="Plotting Ground Truth data", spinner="circle") as spinner:
        gt_data_array_list = load_satellite_dir(train_dir, [SatelliteType.GT])
        gt_data_set = create_satellite_dataset_list(
            gt_data_array_list,
            [SatelliteType.GT],
            list_of_preprocess_func_list=[None],
        )[0]
        gt_tile1_data_array = gt_data_set["Tile1"]
        plot_gt(gt_tile1_data_array, image_dir=save_plots_dir)
        plot_gt_histogram(gt_data_set, image_dir=save_plots_dir)
        spinner.stop_and_persist(
            symbol="✅", text="Finished plotting Ground Truth data"
        )
