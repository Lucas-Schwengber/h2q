# Download MS COCO dataset from the fiftyone package
# The following paper may be useful to understand MS COCO dataset
# https://arxiv.org/abs/1405.0312
# And here is the documentation for the integration between fiftyone and MS COCO
# https://docs.voxel51.com/integrations/coco.html 

import fiftyone as fo
import fiftyone.zoo as foz
import pathlib

max_samples = 120000

# Make output directory if missing
OUTPUT_DIR = pathlib.Path("data/raw/MS_COCO")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = foz.load_zoo_dataset(
    "coco-2014",
    splits=["train","validation"],
    max_samples=max_samples,
    shuffle=False,
    dataset_dir = OUTPUT_DIR
)

# You can vizualize the downloaded images and annotations using the fiftyone API:
# session = fo.launch_app(dataset)

# breakpoint()
