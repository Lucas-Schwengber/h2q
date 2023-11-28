# Download and process cifar dataset
# The following papers may be useful to understand cifar dataset and its hashing applications
# https://arxiv.org/pdf/1802.02904.pdf
# https://openaccess.thecvf.com/content_cvpr_2015/papers/Lai_Simultaneous_Feature_Learning_2015_CVPR_paper.pdf
# https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
# https://ojs.aaai.org/index.php/AAAI/article/download/8952/8811

import pathlib
import tarfile
import urllib.request

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# Make output directory if missing
OUTPUT_DIR = pathlib.Path("data/raw/CIFAR_10")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download tar.gz file
url_stream = urllib.request.urlopen(CIFAR_URL)
tar = tarfile.open(fileobj=url_stream, mode="r|gz")
tar.extractall(path=OUTPUT_DIR)
