# SMART PARKING AI

An image-only Streamlit application that detects cars, motorcycles, buses, and
trucks with pretrained YOLOv8 Nano and classifies marked parking regions as
occupied or empty. It does not access cameras or process videos.

## Requirements

- Python 3.10–3.12 recommended
- Internet access on the first run so Ultralytics can download `yolov8n.pt`
- CUDA-enabled PyTorch/NVIDIA drivers for optional GPU inference; CPU inference
  works without a CUDA GPU

## Setup on Windows

Open PowerShell in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. On the first analysis, Ultralytics downloads
YOLOv8 Nano to `models/yolov8n.pt`. The weights are intentionally not committed;
they are downloaded automatically when needed.

## Run from the Jupyter notebook

Open `SMART_PARKING_AI.ipynb` in VS Code, select a Python 3.10–3.12 kernel, and
choose **Run All**. The notebook installs the requirements, recreates the app in
a `parking_detector/` folder beside the notebook, and starts the web app. It
chooses an available localhost port if the default port is already in use.

## Use

1. Select **Upload Parking Image** and upload a JPG, JPEG, or PNG parking-lot
   image.
2. The app detects vehicles and proposes parking-space outlines where possible.
   Review proposed outlines; they are not guaranteed to match every lot.
3. If proposals are missing or inaccurate, draw a rectangle for each parking
   stall on the image. Name the view and select **Save drawn spaces** to reuse
   its layout on later images from the same viewpoint.
4. Adjust the vehicle overlap threshold and analyze. A space is occupied when a
   detected vehicle's bounding box overlaps at least that fraction of the
   parking region.
5. View results and analytics, download the annotated image, or export detection
   history as CSV.

The total-space count is based on the parking-space regions used in that
analysis, never inferred from vehicle counts. Detection history and saved
layouts are stored in `database/parking_system.db`; annotated images are stored
in `outputs/`.

## Project layout

```text
app.py
config.py
requirements.txt
SMART_PARKING_AI.ipynb
modules/
pages/
assets/style.css
models/                 # YOLOv8 Nano weights download on first analysis
uploads/
outputs/
database/               # SQLite database is created on first launch
```

The pretrained detector is not custom-trained. Model accuracy depends on image
quality, vehicle visibility, and accurate stall boundaries.
