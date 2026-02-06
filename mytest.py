import ptlflow

import cv2 as cv
from google.colab.patches import cv2_imshow

# Download two images to serve as inputs to the optical flow model
# The images below are from the MPI-Sintel dataset: http://sintel.is.tue.mpg.de/
!wget https://github.com/zhou1013mm/sam3/blob/main/run1_000.jpg
!wget https://github.com/zhou1013mm/sam3/blob/main/run1_001.jpg
cv2_imshow(cv.imread("run1_000.jpg"))
cv2_imshow(cv.imread("run1_001.jpg"))

ptlflow.download_scripts()

# If you want to download the script directly from a terminal, you can run:
# python -c "import ptlflow; ptlflow.download_scripts()"

# Go to the folder where the scripts were downloaded to
%cd ptlflow_scripts

!python infer.py --model raft_small --ckpt_path things --input_path ../run1_000.jpg ../run1_001.jpg

# Let's visualize the predicted flow
flow_pred = cv.imread("outputs/inference/raft_small_things/run1_001.jpg")
cv2_imshow(flow_pred)

# Additional dependencies for this example
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils    

# Load the two images
img1 = cv.imread("../run1_000.jpg")
img2 = cv.imread("../run1_001.jpg")

# Get an initialized model from PTLFlow
model = ptlflow.get_model("raft_small", ckpt_path="things")
model.eval()

# IOAdapter is a helper to transform the two images into the input format accepted by PTLFlow models
io_adapter = IOAdapter(model, img1.shape[:2])
inputs = io_adapter.prepare_inputs([img1, img2])