# SENTRA

SENTRA is a web application designed to overcome the persistent challenge of cloud-obscured data in Sentinel-2 multispectral satellite imagery. SENTRA leverages spatiotemporal data extraction and Elastic Net Regression machine learning algorithms to intelligently gap-fill missing pixels. The system automates model training to evaluation and the final synthesis of high-resolution, cloud-free geographic outputs.

## Features

* **Spatiotemporal Model Training:** Train Elastic Net models using historical time-series data with customizable sliding windows and hyperparameters.
* **Model Evaluation:** Validate predictions against Out-of-Time data using statistical (R², MSE, RMSE) and image quality (PSNR, SSIM) metrics.
* **Cloud-Free Reconstruction:** Seamlessly gap-fill cloud-masked areas across 6 Sentinel-2 spectral bands (10m & 20m resolution).
* **Dual-Mode Visualization:** Explore outputs via interactive WebGIS maps (Folium) or export publication-ready transparent PNGs.
* **Flexible Data Inputs:** Use the built-in DVC database (Ciliwung Watershed) or upload custom GeoTIFF targets and Shapefile boundaries.
* **GeoTIFF Export:** Download fully georeferenced reconstructed datasets (`.tif` or `.zip`) ready for GIS software.

## Tech Stack

* **Language:** Python 3.9+
* **Frontend/UI:** Streamlit
* **Machine Learning:** Scikit-Learn (Elastic Net)
* **Geospatial Processing:** GDAL, Rasterio, Geopandas
* **Visualization:** Folium (Interactive), Matplotlib (Static)
* **Deployment & Versioning:** Docker, DVC (Data Version Control)

## Installation Guide
SENTRA provides two installation methods depending on your technical preference: an isolated container architecture via Docker or a manual setup via the GitHub source code.

### Option A: Installation via Docker (Recommended)
This is the most stable and secure method as it pre-configures all geospatial libraries (such as GDAL and Rasterio) inside the container, avoiding any dependency conflicts.
1. **Verify Docker Desktop is running:**
```bash
docker --version
```
2. **Pull the Docker Image:**

```bash
docker pull boviliusbm/sentra:latest
```
3. **Run the Container:**
```bash
docker run -d -p 8501:8501 --name sentra boviliusbm/sentra:latest
```
4. **Access the Interface:**
Open your web browser and navigate to http://localhost:8501.

**Terminate the System:**
```Bash
docker stop sentra
```

### Option B: Installation via GitHub (Source Code)
For researchers or developers who wish to directly modify the source code, the system can be executed using a local Python environment.

1. **Clone the Repository:**
```Bash
git clone https://github.com/BoviliusMeidi/sentra.git
```

2. **Navigate to the Directory:**
```Bash
cd sentra
```

3. **Create a Virtual Environment:**
```Bash
python -m venv venv
```
4. **Activate the Virtual Environment:**

For Windows:
```Bash
venv\Scripts\activate
```
For macOS / Linux:
```Bash
source venv/bin/activate
```
5. **Install Dependencies:**

```Bash
pip install -r requirements.txt
```
6. **Download the Spatial Database (DVC)**:
(This step is mandatory to download the initial satellite imagery datasets to your local folder).

```Bash
dvc pull
```
7. **Run the Application**:
```Bash
streamlit run app.py
```
8. **Access the Interface:**
Open your web browser and navigate to http://localhost:8501.

**Terminate the System:**
Press Ctrl + C in your terminal.
