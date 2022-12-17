import glob
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import geopandas as gpd
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='prepare_ground_truths.py', description='Prepare ground truths')
    parser.add_argument(
        '-ground-truth-claims-path', action='store', help='Insurance claims shapefile filepath', required=True
    )
    args = parser.parse_args()
    print(args)

    ground_truth_claims = pd.concat(Parallel(n_jobs=5, max_nbytes=None)(
        delayed(gpd.read_file)(shp_filepath) for shp_filepath in tqdm(
            glob.glob(args.ground_truth_claims_path + "/*.shp")
        )
    ))

    print(ground_truth_claims.groupby(by='damage_ext').count()['t_dmg_bldg'])
    print(ground_truth_claims.shape)
