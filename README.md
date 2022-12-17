# Flood-Damage-Extent-Detection

Flood damage extent detection using meta attributes

## Insurance Building Claim Stats
``` python
python -W ignore::UserWarning -W ignore::DeprecationWarning ground_truth_stats.py -ground-truth-claims-path E:/SFU/dataset/raw/ground_truth/shapefiles
```
### Stats
| __Damage Extent Level__ | __No. of buildings__  |
|:-----------------------:|:-----------------:|
|          High           |        114        |
|         Medium          |        75         |
|           Low           |        69         |
|        __TOTAL__        |      __258__      |

&nbsp;

## How to generate ground truths
### Parallel processing
``` python
python -W ignore::UserWarning -W ignore::DeprecationWarning prepare_ground_truths.py -mask-img-path <mask-img-path> -pre-img-path <pre-img-path> -aoi-tif-path <aoi-tif-path> -claims-filepath <claims-filepath> -ground-truth-path <ground-truth-path> -parallel
```
### Sequential processing
``` python
python -W ignore::UserWarning -W ignore::DeprecationWarning prepare_ground_truths.py -mask-img-path <mask-img-path> -pre-img-path <pre-img-path> -aoi-tif-path <aoi-tif-path> -claims-filepath <claims-filepath> -ground-truth-path <ground-truth-path>
```
