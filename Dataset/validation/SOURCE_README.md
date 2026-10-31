# NYC test set: traffic cones and construction signs

300 street-level images from the five NYC boroughs, from the Mapillary API (1024 px thumbnails),
built 2026-10-30 by build_datasets.py. Imagery (c) Mapillary contributors, CC BY-SA 4.0.

| group | images |
|---|---|
| cone_only | 100 |
| sign_only | 100 |
| both | 40 |
| negative | 60 |

Filters: no panoramas, one image per capture sequence, captured 2018 or later, at least 20
Mapillary detections, inside an NYC borough per reverse geocoding.

Sampling: construction signs appear in under 0.5% of random NYC images, so the sign_only and both groups come
from images of signs on Mapillary's map (map_features endpoint, one image per sign). cone_only and negative
come from random ~400 m boxes across the five boroughs.

The images are split across 2 zips to stay under 25 MB each: part 1 = cone_only + negative, part 2 = sign_only + both. Unzip all parts into the
same folder; each one carries the full CSVs and this README.

`images.csv` has location, capture date and a link for each image. `annotations.csv` has one box per
cone/sign from **Mapillary's segmentation model**: these are predictions, so they were checked by hand
(see nyc_preview/ while working) before being used as ground truth.
Sign labels counted: ['object--traffic-sign--temporary-front'] plus any value starting with ['warning--roadworks'].
