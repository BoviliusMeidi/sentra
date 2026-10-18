import logging
import rasterio
from rasterio.mask import mask
import geopandas as gpd

class VectorClipper:
    """
    Object-Oriented module for clipping raster imagery using vector boundaries (Shapefile).
    Equipped with dynamic Coordinate Reference System (CRS) transformation and robust parsing.
    """
    def __init__(self, shapefile_path: str):
        self.shapefile_path = shapefile_path
        try:
            self.gdf = gpd.read_file(shapefile_path)

            if self.gdf.crs is None:
                logging.warning("⚠️ Shapefile CRS is undefined (naive geometry). Explicitly setting default to EPSG:4326 (WGS 84).")
                self.gdf = self.gdf.set_crs("EPSG:4326", allow_override=True)

            logging.info(f"✅ Shapefile loaded successfully: {shapefile_path} (CRS: {self.gdf.crs})")
        except Exception as e:
            raise RuntimeError(f"Failed to read the Shapefile. Ensure all required companion files (.shp, .shx, .dbf, .prj) are uploaded together. Details: {e}")

    def apply(self, src: rasterio.DatasetReader, meta: dict) -> tuple:
        """
        Performs mathematical CRS alignment and executes the geometric raster crop.
        """
        raster_crs = src.crs

        if self.gdf.crs != raster_crs:
            logging.info(f"🔄 Reprojecting Shapefile CRS from {self.gdf.crs} to {raster_crs}...")
            gdf_projected = self.gdf.to_crs(raster_crs)
        else:
            gdf_projected = self.gdf

        shapes = [feature["geometry"] for _, feature in gdf_projected.iterrows()]

        try:
            out_image, out_transform = mask(src, shapes, crop=True)
            meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            return out_image, meta
        except ValueError as e:
            logging.error(f"❌ Spatial clipping failed: {e}")
            raise ValueError("Clipping execution failed: The Shapefile geometry and Raster imagery are spatially disjoint. Verify that the uploaded TIFF files geographically intersect the selected vector polygon.")