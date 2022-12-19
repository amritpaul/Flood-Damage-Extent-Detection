import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import warnings
import imageio
import rasterio as rio
from rasterio import features
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon, mapping
import pprint
from collections import namedtuple
from pathlib import Path
import joblib
import matplotlib.patches as mpatches
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import argparse

gpd.options.display_precision = 6
image_boundary_offset = 50


def img_to_geo_coord(point, bounds, img_shape, offset):
    return (bounds.left + ((point[0] - offset[0]) / img_shape[0]) * (bounds.right - bounds.left)), (
            bounds.top - ((point[1] - offset[1]) / img_shape[1]) * (bounds.top - bounds.bottom))


def get_bounds(img_count, orignial_h, orignial_w, orignial_bounds, original_res, img_shape):
    named_tuple_bound = namedtuple('BoundingBox', {'left': 0, 'right': 0, 'bottom': 0, 'top': 0})
    cnt = 0
    img_count -= 1
    p = 0
    for i in range(0, orignial_h, img_shape[1]):
        for j in range(0, orignial_w, img_shape[0]):
            if (cnt == img_count):
                p = 1
                break
            cnt += 1
        if p:
            break
    return named_tuple_bound(**{
        'left': orignial_bounds.left + j * original_res[0],
        'right': orignial_bounds.left + (j + img_shape[0]) * original_res[0],
        'bottom': orignial_bounds.top - (i + img_shape[1]) * original_res[1],
        'top': orignial_bounds.top - (i * original_res[1])
    })


def geo_to_img_coord(point, bounds, img_shape):
    return (((point[0] - bounds.left) / (bounds.right - bounds.left)) * img_shape[0],
            ((bounds.top - point[1]) / (bounds.top - bounds.bottom)) * img_shape[1])


def merge_claims_with_polygons(claims, polygons, ground_truth_path, img_mask_path):
    g = gpd.sjoin(polygons, claims.to_crs(crs=32615), how='left', predicate='contains')
    g.dropna(subset=['damage_ext'], inplace=True)
    if g.shape[0] == 0:
        return None
    g.loc[g['damage_ext'] == 'low', 'damage_extent_class_index'] = 1
    g.loc[g['damage_ext'] == 'medium', 'damage_extent_class_index'] = 2
    g.loc[g['damage_ext'] == 'high', 'damage_extent_class_index'] = 3

    g['damage_extent_class_index'] = g['damage_extent_class_index'].astype('int8')

    Path(ground_truth_path.joinpath('shapefiles')).mkdir(parents=True, exist_ok=True)
    g.to_file(ground_truth_path.joinpath('shapefiles').joinpath(f"{img_mask_path.stem}.shp"))
    return g


def get_geo_coords(shapes, bounds):
    coords = []
    for polygon in shapes['coordinates']:
        coords.append(Polygon(list(map(lambda point: img_to_geo_coord(
            point, bounds, (1024, 1024), offset=(image_boundary_offset, image_boundary_offset)), polygon))))
    return coords


def get_img_coords(merged_claims, bounds):
    coords_reversed = []
    for i in range(merged_claims.shape[0]):
        coords_reversed.append((
            list(map(lambda point: geo_to_img_coord(point, bounds, (1024, 1024)),
                     mapping(merged_claims['geometry'].iloc[i])['coordinates'][0])
                 ), merged_claims.damage_extent_class_index.iloc[i]))
    return coords_reversed


def get_ground_truth_mask(new_coords_reversed, output_shape, transform, ground_truth_filepath):
    mask = features.rasterize(
        shapes=list(map(lambda x: ({'type': 'Polygon', 'coordinates': [x[0]]}, x[1]), new_coords_reversed)),
        out_shape=output_shape, transform=transform)
    ground_truth_mask = np.dstack((mask, mask, mask)).astype('uint8')
    ground_truth_mask[mask == 1] = (1, 1, 1)
    ground_truth_mask[mask == 2] = (2, 2, 2)
    ground_truth_mask[mask == 3] = (3, 3, 3)
    imageio.imsave(ground_truth_filepath, ground_truth_mask)
    return ground_truth_mask, mask


def plot_overlap_with_raw(pre_img_filepath, ground_truth_path, ground_truth_mask, mask):
    pre_img = imageio.imread(pre_img_filepath)
    ground_truth_mask[mask == 1] = (0, 255, 0)
    ground_truth_mask[mask == 2] = (0, 0, 255)
    ground_truth_mask[mask == 3] = (255, 0, 0)

    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_axis_off()
    ax1.imshow(pre_img)
    ax1.imshow(ground_truth_mask, alpha=0.3)

    low_patch = mpatches.Patch(color='green', label='Low damage')
    medium_patch = mpatches.Patch(color='blue', label='Medium damage')
    high_patch = mpatches.Patch(color='red', label='High damage')

    plt.legend(handles=[low_patch, medium_patch, high_patch], loc="upper left", prop={'size': 20})
    Path(ground_truth_path.joinpath('overlap_with_raw')).mkdir(parents=True, exist_ok=True)
    fig.savefig(ground_truth_path.joinpath(f'overlap_with_raw/{pre_img_filepath.stem}.png'), bbox_inches='tight')
    plt.close("all")


def process_image(claims_data, pre_img_filepath, mask_img_filepath, aoi_tif_path, ground_truth_path):
    cnt = 0
    src = rio.open(mask_img_filepath)
    image = np.pad(src.read(), pad_width=(
        (0, 0), (image_boundary_offset, image_boundary_offset), (image_boundary_offset, image_boundary_offset)
    ), constant_values=0)
    shapes = list(features.shapes(source=image, mask=image != 255, transform=src.transform))[-1][0]

    area_tif = rio.open(aoi_tif_path.joinpath(f'{mask_img_filepath.stem.split("_")[0]}_pre.tif'))
    img = area_tif.read()
    bounds = get_bounds(
        int(mask_img_filepath.stem.split("_")[-1]),
        area_tif.height, area_tif.width, area_tif.bounds, area_tif.res, (1024, 1024)
    )
    new_coords = get_geo_coords(shapes, bounds)

    polygons = gpd.GeoDataFrame({'geometry': new_coords}, crs="EPSG:32615").drop([0])
    polygons.index = list(range(polygons.shape[0]))
    merged_claims = merge_claims_with_polygons(claims_data, polygons, ground_truth_path, mask_img_filepath)
    if merged_claims is not None:
        new_coords_reversed = get_img_coords(merged_claims, bounds)
        ground_truth_mask, mask = get_ground_truth_mask(
            new_coords_reversed, src.shape, src.transform, ground_truth_path.joinpath(mask_img_filepath.name)
        )
        plot_overlap_with_raw(
            pre_img_filepath=pre_img_filepath, ground_truth_path=ground_truth_path, ground_truth_mask=ground_truth_mask,
            mask=mask
        )
        cnt = 1
    return cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='prepare_ground_truths.py', description='Prepare ground truths')
    parser.add_argument(
        '-claims-filepath', dest='claims_filepath', action='store', help='Insurance claims shapefile filepath',
        required=True
    )
    parser.add_argument('-mask-img-path', action='store', help='Image mask path', required=True)
    parser.add_argument('-pre-img-path', action='store', help='Pre-image path', required=True)
    parser.add_argument('-aoi-tif-path', action='store', help='AOI TIF path', required=True)
    parser.add_argument('-ground-truth-path', action='store', help='Ground truth path', required=True)
    parser.add_argument('-parallel', action='store_true')

    args = parser.parse_args()

    path = Path(r'E:/SFU/dataset/raw/temp')

    claims = gpd.read_file(Path(args.claims_filepath))
    result_count = 0

    if args.parallel:
        result = Parallel(n_jobs=5, max_nbytes=None)(
            delayed(process_image)(
                claims_data=claims, aoi_tif_path=Path(args.aoi_tif_path),
                ground_truth_path=Path(args.ground_truth_path),
                pre_img_filepath=Path(args.pre_img_path).joinpath(img_name),
                mask_img_filepath=Path(args.mask_img_path).joinpath(img_name),
            ) for img_name in tqdm(os.listdir(Path(args.mask_img_path).as_posix()))
        )
        result_count = sum(result)
    else:
        for img_name in tqdm(os.listdir(Path(args.mask_img_path).as_posix())):
            result_count += process_image(
                claims_data=claims, aoi_tif_path=Path(args.aoi_tif_path),
                ground_truth_path=Path(args.ground_truth_path),
                pre_img_filepath=Path(args.pre_img_path).joinpath(img_name),
                mask_img_filepath=Path(args.mask_img_path).joinpath(img_name),
            )
    print(f"\nTotal number of ground truth masks: {result_count}. Open: {Path(args.ground_truth_path)}")
