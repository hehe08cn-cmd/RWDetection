"""Global configuration for runway detector."""
import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Video files
VIDEOS = {
    "video1": os.path.join(DATA_DIR, "A330 - 2025-06-04 11.05.05.mp4"),
    "video2": os.path.join(DATA_DIR, "A330 - 2025-06-05 01.14.58.mp4"),
    "video3": os.path.join(DATA_DIR, "A330 - 2025-06-05 10.05.15.mp4"),
}

# Pose files
POSES = {
    "video1": os.path.join(DATA_DIR, "A330 - 2025-06-04 11.05.05.txt"),
    "video2": os.path.join(DATA_DIR, "A330 - 2025-06-05 01.14.58.txt"),
    "video3": os.path.join(DATA_DIR, "A330 - 2025-06-05 10.05.15.txt"),
}

# Train/test split
TRAIN_VIDEOS = ["video1", "video2", "video3"]
TEST_VIDEOS = ["video3"]

# video3 frame split for cross-validation (avoid pitch angle domain shift)
# Training uses frames before this index, validation/testing uses frames >= this index
VIDEO3_SPLIT_FRAME = 1250

# Image
ORIGINAL_SIZE = (1920, 1080)
WORKING_SIZE = (512, 288)  # (width, height) - 1/4 of original
CROP_SCALE = 3.0  # crop region = 3x corner bounding box

# Model
NUM_CORNERS = 4
BACKBONE = "hrnet_w18"  # from timm
PRETRAINED = True
INPUT_CHANNELS = 3  # RGB only in Stage 1; Stage 3 adds 7ch prior → 10ch total

# HRNet feature extraction (multi-scale fusion matching detection project pattern)
# Extract all 5 stages, skip stride-2 (feat[0]), upsample+concat lower stages to 1/4
# timm hrnet_w18_small_v2: feat[1]=128ch(1/4) + feat[2]=256ch(1/8) + feat[3]=512ch(1/16) + feat[4]=1024ch(1/32)
HRNET_OUT_INDICES = (0, 1, 2, 3, 4)  # all stages, fused in backbone.forward()
HRNET_FEATURE_CHANNELS = [64, 128, 256, 512, 1024]  # per-stage channel counts
HRNET_FUSION_CHANNELS = 256  # 1x1 projection after multi-scale concat (reduces 1920ch → 256ch)

# Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
FRAME_STRIDE = 5  # sample every 5th frame for training
VAL_STRIDE = 50  # sample every 50th frame for validation
NUM_WORKERS = 4

# Heatmap
HEATMAP_SIGMA = 3.0  # Gaussian sigma for corner GT heatmaps (pixels at working resolution)

# Loss weights
LOSS_WEIGHTS = {
    "heatmap": 1.0,
    "coord": 0.1,
    "edge": 0.5,
    "centerline": 0.5,
    "geom": 0.05,
    "temporal": 0.2,
}

# Mode switching
FAR_FIELD_ALTITUDE_THRESHOLD = 30.0  # meters AGL
NEAR_FIELD_ALTITUDE_THRESHOLD = 30.0

# Inference
INFERENCE_BATCH_SIZE = 1
MC_DROPOUT_SAMPLES = 5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
