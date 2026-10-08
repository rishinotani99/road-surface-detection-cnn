# Road Surface Condition Detection Using CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rishinotani99/road-surface-detection-cnn/blob/main/notebooks/Road_Surface_Detection_CNN.ipynb)

A deep-learning system that looks at a road photo and classifies it as **normal** or **pothole**, shows how confident it is, and draws a **Grad-CAM heatmap** of the area the model focused on.

- **Training:** Google Colab (free T4 GPU) — `notebooks/Road_Surface_Detection_CNN.ipynb`
- **Web app:** FastAPI backend + single-page HTML frontend, run locally from this repo

---

## How the system works

```
Road photo ──► FastAPI /api/predict ──► MobileNetV2 (transfer learning) ──► label + confidence + probabilities
                                                    └─ last feature maps (out_relu) ──► Grad-CAM heatmap overlay (PNG)
                                                                                        │
Browser (frontend/index.html) ◄────────────────────── JSON result ◄─────────────────────┘
```

1. **Dataset** – the Kaggle *Pothole Detection Dataset* (`atulyakumar98/pothole-detection-dataset`), two classes: `normal` and `potholes`. Downloaded automatically with `kagglehub`; unreadable images are removed.
2. **Split** – stratified 70% train / 15% validation / 15% test, images resized to 224×224.
3. **Augmentation** – random flip, rotation, zoom, contrast and brightness (training only).
4. **Model A – Custom CNN** – 4 conv blocks (Conv → BatchNorm → ReLU → MaxPool), GlobalAveragePooling, Dropout, softmax. Trained from scratch with early stopping, followed by a sanity check that warns if it only learned to predict one class.
5. **Model B – MobileNetV2** – ImageNet-pretrained base frozen while a new head is trained, then the top ~30 layers are fine-tuned with a low learning rate.
6. **Evaluation** – test accuracy, classification report, confusion matrix and a comparison chart.
7. **Explainability** – Grad-CAM highlights the image regions behind each decision. The notebook shows it for both models: the custom CNN's `last_conv` layer and MobileNetV2's `out_relu` feature maps.
8. **Deployment** – the backend uses MobileNetV2 (the more accurate model) both for the prediction and for the Grad-CAM heatmap, so the heatmap always explains the label shown. The custom CNN is the from-scratch baseline used for the comparison in the notebook.

## Folder structure

```
road-surface-detection-cnn/
├── notebooks/Road_Surface_Detection_CNN.ipynb   # training notebook (run in Colab)
├── backend/app.py                               # FastAPI app (API + serves the frontend)
├── frontend/index.html                          # upload page (HTML/CSS/JS)
├── models/                                      # trained .keras files + classes.json go here
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Train the model in Google Colab

1. Click **Open in Colab** at the top of this README (or open https://colab.research.google.com → **File → Open notebook → GitHub** and pick this repo).
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** The dataset downloads automatically — no Kaggle login needed for this public dataset. Training takes roughly 10–20 minutes on a T4.
4. The last cell downloads three files:
   - `road_surface_mobilenetv2.keras`
   - `road_surface_custom_cnn.keras`
   - `classes.json`

   If the browser blocks them, allow multiple downloads for colab.research.google.com, or get them from the Colab file panel (📁 → `models/`).
5. Note the **TensorFlow version** printed by the last cell.
6. To save notebook edits back to GitHub: **File → Save a copy in GitHub**, pick this repo, branch `main`, path `notebooks/Road_Surface_Detection_CNN.ipynb`.

Optional: to test your own photo inside Colab, set `UPLOAD_IMAGE = True` in section 10 and re-run that cell (it's `False` by default so *Run all* never waits for input).

## 2. Put the model files in `models/`

Move the three downloaded files into the `models/` folder:

```
models/
├── road_surface_mobilenetv2.keras   # required — prediction + Grad-CAM heatmap
├── road_surface_custom_cnn.keras    # baseline model from the comparison (not used by the app)
└── classes.json                     # class names in order, e.g. ["normal", "potholes"]
```

If `classes.json` is missing, the app falls back to `["normal", "potholes"]`.

Then commit them so your whole group has them:

```bash
git add models/
git commit -m "Add trained models"
git push
```

## 3. Run the app locally

Requires **Python 3.10 – 3.13** (TensorFlow doesn't support 3.14 yet).

```powershell
# Windows (PowerShell) — first time only
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# every time
.\venv\Scripts\Activate.ps1
uvicorn backend.app:app --reload
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Open **http://127.0.0.1:8000**, drop in a road photo and click **Analyse road**.
Interactive API docs: http://127.0.0.1:8000/docs

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | The web page (`frontend/index.html`) |
| `GET` | `/api/health` | Server status, class names, whether the model and Grad-CAM are loaded |
| `POST` | `/api/predict` | Upload an image (`multipart/form-data`, field `file`) → prediction |

Example:

```bash
curl -F "file=@road.jpg" http://127.0.0.1:8000/api/predict
```

```json
{
  "label": "potholes",
  "is_pothole": true,
  "confidence": 0.97,
  "probabilities": { "normal": 0.03, "potholes": 0.97 },
  "advice": "Pothole detected. Drive slowly and steer around it if safe, and report this location to the local road maintenance authority.",
  "gradcam": "data:image/png;base64,iVBORw0KGgo...",
  "gradcam_available": true
}
```

Errors: `400` for files that aren't images, `503` if the model isn't loaded (the message explains what to do).

## Troubleshooting

**"Model file not found: models/road_surface_mobilenetv2.keras"**
The model hasn't been trained/copied yet. Run the Colab notebook, download the three files, put them in `models/` (exact filenames above) and restart `uvicorn`.

**"Could not load models/… " / errors mentioning Keras, `deserialize`, or unknown layers — TensorFlow version mismatch**
Models saved with one Keras version may not load in a much older one. Check both versions:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```
and compare with the version printed at the end of the notebook. Install the matching one, e.g. `pip install "tensorflow==2.19.*"`. If you can't match exactly, local should be the **same or newer** than Colab's.

**No GPU in Colab (`GPU available: False`)**
Runtime → Change runtime type → **T4 GPU** → Save, then Runtime → Restart and run all. Free GPU time is limited; if none is available, wait a while or train on CPU (it works, just slower — you can reduce epochs).

**Grad-CAM heatmap doesn't appear**
Check `/api/health` → `"gradcam_available"` and the server log at startup (`Grad-CAM uses layer …` or `Grad-CAM disabled — <reason>`). The app still predicts without it. The heatmap is computed from MobileNetV2's `out_relu` layer, so it needs a model built like the one in the notebook.

**Custom CNN accuracy is ~50% / it predicts one class for everything**
Training collapsed. The notebook prints a ⚠️ warning after section 5 when this happens — re-run the build and train cells of section 5 (and everything after it). The web app isn't affected, since it uses MobileNetV2.

**Server crashes, or "The paging file is too small" when starting**
TensorFlow needs roughly 1–1.5 GB of free RAM. Close Chrome tabs and other heavy apps, and don't run two TensorFlow processes at once.

**The page says "Backend offline"**
Start the server from the project root (`uvicorn backend.app:app --reload`) with the venv activated, and open http://127.0.0.1:8000.

**`Activate.ps1 cannot be loaded because running scripts is disabled`**
Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Model files bigger than 100 MB**
GitHub rejects files over 100 MB. Our models are much smaller (~10–30 MB), but if one ever grows past that, use Git LFS:
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes models/
git commit -m "Track .keras files with Git LFS"
git push
```

## Future scope

- **More classes** – cracks, wet/flooded roads, speed bumps, gravel, snow/ice.
- **Video input** – classify dash-cam video frame by frame in real time.
- **GPS mapping** – tag each detection with coordinates and plot potholes on a map for road authorities.
- **Scan history** – store every scan (image, result, time, location) in an SQLite database with a history page.
- **Object detection** – use YOLO-style models to draw a box around each pothole and estimate its size.
- **Mobile app** – convert the model to TensorFlow Lite for on-device detection.
