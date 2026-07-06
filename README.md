# Wildfire Risk Prediction from Multi-Band Satellite Imagery

End-to-end pipeline for pixel-level wildfire risk prediction using spatio-temporal deep learning on multi-band satellite imagery. The study area is mainland Portugal, one of the most fire-affected regions in Europe.

---

## Pipeline Overview

| Module | Description |
|--------|-------------|
| `src/gee_utils.py` | Google Earth Engine functions for building bi-weekly composites, burn labels, and streaming patches via `computePixels` |
| `src/sampling.py` | Stratified fire/non-fire point sampling with spatial block train/val split |
| `src/dataset.py` | PyTorch `IterableDataset` for streaming temporal sequences directly from GEE |
| `notebooks/wildfire_risk_pipeline.ipynb` | Full walkthrough — data visualisation, temporal analysis, model training, and prediction |

**Band stack (10 bands):** Sentinel-2 RGB + SWIR1/SWIR2, NDVI, NDWI, MODIS LST, SRTM Elevation, Slope

**Architecture:** ConvLSTM encoder + convolutional decoder for sequence-to-one pixel-level segmentation. Designed to scale to U-TAE for production use.

---

## Setup and Usage

**Requirements:** Python 3.10+, a [Google Earth Engine](https://earthengine.google.com/) account, and a GEE project ID.

```bash
git clone https://github.com/scossey/Fire_risk_prediction
cd Fire_risk_prediction
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
EE_PROJECT_ID=your-gee-project-id
```

The notebook provides a full walkthrough.
