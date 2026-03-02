"""
Read common vector layer formats and extract hyperspectral signatures beneath vector features.
"""

import os
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio import features
from rasterio.transform import Affine
from typing import Union, Tuple, Optional, List
from tqdm import tqdm

import BlackTelperion
from BlackTelperion import BlackImage, BlackLibrary


def load_vector_file(path):
    """
    Loads a vector file into a GeoDataFrame using geopandas.

    Supports common vector formats: .shp, .gpkg, .geojson, .kml, etc.

    Args:
        path (str): Path to vector file (.shp, .gpkg, .geojson, etc.)

    Returns:
        geopandas.GeoDataFrame: Loaded vector layer with geometries and attributes

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector file not found: {path}")

    try:
        gdf = gpd.read_file(path)
        return gdf
    except Exception as e:
        raise ValueError(f"Failed to load vector file {path}: {str(e)}")


def extract_signatures_from_vector(
    image: BlackImage,
    vector_path: str,
    output_csv: Optional[str] = None,
    neighbor_size: int = 1,
    aggregate: Union[bool, str] = False,
    label_field: Optional[str] = None,
    return_format: str = 'dataframe',
    progress: bool = True
) -> Union[pd.DataFrame, BlackLibrary, Tuple[pd.DataFrame, BlackLibrary]]:
    """
    Extract spectral signatures from hyperspectral image based on vector geometries.

    - For **polygons**: Extracts all pixels beneath the polygon
    - For **points**: Extracts the pixel beneath + N Moore neighbors (NxN window)

    Args:
        image (BlackImage): BlackImage instance containing hyperspectral data
        vector_path (str): Path to vector layer file (.shp, .gpkg, .geojson, etc.)
        output_csv (str, optional): Path to save CSV output. If None, no CSV is written
        neighbor_size (int): For point features, N where window is (2*N+1) x (2*N+1).
                           Default=1 creates 3x3 window. N=2 creates 5x5 window, etc.
        aggregate (bool or str): How to aggregate pixels per feature:
            - False or 'all': Return all pixels individually (default)
            - 'mean': Average spectra per feature
            - 'median': Median spectra per feature
            - 'std': Standard deviation per feature
            - 'percentile_X': Xth percentile (e.g., 'percentile_50' for median)
        label_field (str, optional): Name of vector attribute containing class labels.
                                    If None, uses feature IDs
        return_format (str): Output format - 'dataframe', 'blacklibrary', or 'both'
        progress (bool): Show progress bar. Default is True

    Returns:
        pd.DataFrame, BlackLibrary, or tuple depending on return_format:
            - 'dataframe': Returns pandas DataFrame
            - 'blacklibrary': Returns BlackLibrary instance
            - 'both': Returns tuple (DataFrame, BlackLibrary)

    Example:
        >>> image = BlackTelperion.io.load('hyperspectral.hdr')
        >>> df = extract_signatures_from_vector(
        ...     image,
        ...     'ground_truth.gpkg',
        ...     output_csv='signatures.csv',
        ...     neighbor_size=2,
        ...     label_field='class_name'
        ... )
    """
    # Load vector file
    gdf = load_vector_file(vector_path)

    # Align coordinate systems
    gdf_aligned = _align_crs(gdf, image)

    # Get wavelengths
    wavelengths = image.get_wavelengths()
    if wavelengths is None:
        wavelengths = np.arange(image.band_count())

    # Extract signatures
    records = []
    iterator = tqdm(gdf_aligned.iterrows(), total=len(gdf_aligned), desc="Extracting signatures") if progress else gdf_aligned.iterrows()

    for idx, feature in iterator:
        feature_id = int(idx)
        label = _get_label(feature, label_field, feature_id)
        geometry = feature.geometry

        if geometry.geom_type in ['Polygon', 'MultiPolygon']:
            pixels = _extract_from_polygon(image, geometry, feature_id, label)
        elif geometry.geom_type in ['Point', 'MultiPoint']:
            pixels = _extract_from_point(image, geometry, feature_id, label, neighbor_size)
        else:
            print(f"Warning: Skipping unsupported geometry type '{geometry.geom_type}' for feature {feature_id}")
            continue

        records.extend(pixels)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    if len(df) == 0:
        raise ValueError("No pixels were extracted. Check that vector geometries overlap with image extent.")

    # Apply aggregation if requested
    if aggregate and aggregate != 'all':
        df = _aggregate_signatures(df, aggregate, wavelengths)

    # Save CSV if requested
    if output_csv is not None:
        _save_csv_with_metadata(df, output_csv, wavelengths)

    # Return in requested format
    if return_format == 'dataframe':
        return df
    elif return_format == 'blacklibrary':
        return _dataframe_to_blacklibrary(df, wavelengths)
    elif return_format == 'both':
        return df, _dataframe_to_blacklibrary(df, wavelengths)
    else:
        raise ValueError(f"Invalid return_format: {return_format}. Must be 'dataframe', 'blacklibrary', or 'both'")


def _align_crs(gdf: gpd.GeoDataFrame, image: BlackImage) -> gpd.GeoDataFrame:
    """
    Align vector CRS to match image CRS.

    Args:
        gdf: GeoDataFrame to reproject
        image: BlackImage containing target CRS

    Returns:
        Reprojected GeoDataFrame
    """
    image_crs = image.projection

    if image_crs is None:
        print("Warning: Image has no CRS defined. Assuming vector and image share the same coordinate system.")
        return gdf

    # Get vector CRS
    vector_crs = gdf.crs

    if vector_crs is None:
        print("Warning: Vector layer has no CRS defined. Assuming it matches the image CRS.")
        return gdf

    # Convert GDAL SpatialReference to WKT or EPSG if needed
    if hasattr(image_crs, 'ExportToWkt'):
        # It's a GDAL SpatialReference object
        try:
            # Try to get EPSG code first
            epsg_code = image_crs.GetAuthorityCode(None)
            if epsg_code:
                image_crs_str = f'EPSG:{epsg_code}'
            else:
                # Fall back to WKT
                image_crs_str = image_crs.ExportToWkt()
        except:
            # If all else fails, use WKT
            image_crs_str = image_crs.ExportToWkt()
    else:
        # Assume it's already a string or compatible format
        image_crs_str = image_crs

    # Reproject if different
    try:
        # Convert to comparable formats
        if str(vector_crs) != str(image_crs_str):
            gdf_aligned = gdf.to_crs(image_crs_str)
            return gdf_aligned
    except Exception as e:
        print(f"Warning: CRS reprojection failed: {e}. Proceeding with original CRS.")

    return gdf


def _get_label(feature, label_field: Optional[str], feature_id: int) -> str:
    """
    Extract label from feature attributes.

    Args:
        feature: GeoDataFrame row
        label_field: Name of attribute field containing label
        feature_id: Fallback numeric ID

    Returns:
        Label as string
    """
    if label_field and label_field in feature.index:
        return str(feature[label_field])
    else:
        return f"Feature_{feature_id}"


def _extract_from_polygon(
    image: BlackImage,
    geometry,
    feature_id: int,
    label: str
) -> List[dict]:
    """
    Extract all pixels beneath a polygon geometry.

    Args:
        image: BlackImage instance
        geometry: Shapely Polygon or MultiPolygon
        feature_id: Numeric feature identifier
        label: Class label string

    Returns:
        List of dictionaries containing pixel data
    """
    records = []

    # Convert affine to rasterio format
    transform = Affine(*image.affine)

    # Create a mask for this polygon
    try:
        # Rasterize the geometry
        geoms = [geometry.__geo_interface__]
        mask_array = features.rasterize(
            geoms,
            out_shape=(image.xdim(), image.ydim()),
            transform=transform,
            all_touched=True,
            dtype=np.uint8
        )

        # Get pixel coordinates where mask is True
        y_indices, x_indices = np.where(mask_array > 0)

        # Extract spectral signatures
        for y, x in zip(y_indices, x_indices):
            if y < image.xdim() and x < image.ydim():  # Bounds check
                spectrum = image.data[y, x, :]

                # Skip if all NaN or all zeros (data ignore)
                if not np.all(np.isnan(spectrum)) and not np.all(spectrum == 0):
                    # Check for data ignore value in header
                    ignore_value = None
                    if 'data ignore value' in image.header:
                        try:
                            ignore_value = float(image.header['data ignore value'])
                        except:
                            pass

                    # Skip if spectrum matches ignore value
                    if ignore_value is not None and np.all(spectrum == ignore_value):
                        continue

                    # Convert pixel to world coordinates
                    world_x, world_y = _pixel_to_world(x, y, transform)

                    record = {
                        'feature_id': feature_id,
                        'feature_name': label,
                        'pixel_x': int(x),
                        'pixel_y': int(y),
                        'world_x': world_x,
                        'world_y': world_y,
                    }

                    # Add spectral bands
                    for band_idx, value in enumerate(spectrum):
                        record[f'band_{band_idx}'] = value

                    records.append(record)

    except Exception as e:
        print(f"Warning: Failed to extract pixels for polygon {feature_id}: {e}")

    return records


def _extract_from_point(
    image: BlackImage,
    geometry,
    feature_id: int,
    label: str,
    neighbor_size: int
) -> List[dict]:
    """
    Extract pixel beneath point + N Moore neighbors (creates (2N+1)x(2N+1) window).

    Args:
        image: BlackImage instance
        geometry: Shapely Point or MultiPoint
        feature_id: Numeric feature identifier
        label: Class label string
        neighbor_size: N where window size is (2*N+1) x (2*N+1)

    Returns:
        List of dictionaries containing pixel data
    """
    records = []

    transform = Affine(*image.affine)

    # Handle MultiPoint
    points = [geometry] if geometry.geom_type == 'Point' else list(geometry.geoms)

    for point in points:
        # Convert world coordinates to pixel coordinates
        px, py = _world_to_pixel(point.x, point.y, transform)

        # Extract window around point
        for dy in range(-neighbor_size, neighbor_size + 1):
            for dx in range(-neighbor_size, neighbor_size + 1):
                y = py + dy
                x = px + dx

                # Bounds check
                if 0 <= y < image.xdim() and 0 <= x < image.ydim():
                    spectrum = image.data[y, x, :]

                    # Skip if all NaN or all zeros (data ignore)
                    if not np.all(np.isnan(spectrum)) and not np.all(spectrum == 0):
                        # Check for data ignore value in header
                        ignore_value = None
                        if 'data ignore value' in image.header:
                            try:
                                ignore_value = float(image.header['data ignore value'])
                            except:
                                pass

                        # Skip if spectrum matches ignore value
                        if ignore_value is not None and np.all(spectrum == ignore_value):
                            continue

                        # Convert back to world coordinates
                        world_x, world_y = _pixel_to_world(x, y, transform)

                        record = {
                            'feature_id': feature_id,
                            'feature_name': label,
                            'pixel_x': int(x),
                            'pixel_y': int(y),
                            'world_x': world_x,
                            'world_y': world_y,
                        }

                        # Add spectral bands
                        for band_idx, value in enumerate(spectrum):
                            record[f'band_{band_idx}'] = value

                        records.append(record)

    return records


def _pixel_to_world(px: int, py: int, transform) -> Tuple[float, float]:
    """Convert pixel coordinates to world coordinates using affine transform."""
    world_x, world_y = transform * (px, py)
    return world_x, world_y


def _world_to_pixel(world_x: float, world_y: float, transform) -> Tuple[int, int]:
    """Convert world coordinates to pixel coordinates using inverse affine transform."""
    inv_transform = ~transform
    px, py = inv_transform * (world_x, world_y)
    return int(round(px)), int(round(py))


def _aggregate_signatures(
    df: pd.DataFrame,
    method: str,
    wavelengths: np.ndarray
) -> pd.DataFrame:
    """
    Aggregate spectral signatures per feature.

    Args:
        df: DataFrame with individual pixel signatures
        method: Aggregation method ('mean', 'median', 'std', 'percentile_X')
        wavelengths: Array of wavelength values

    Returns:
        Aggregated DataFrame with one row per feature
    """
    band_cols = [col for col in df.columns if col.startswith('band_')]
    group_cols = ['feature_id', 'feature_name']

    if method == 'mean':
        agg_df = df.groupby(group_cols)[band_cols].mean().reset_index()
    elif method == 'median':
        agg_df = df.groupby(group_cols)[band_cols].median().reset_index()
    elif method == 'std':
        agg_df = df.groupby(group_cols)[band_cols].std().reset_index()
    elif method.startswith('percentile_'):
        percentile = float(method.split('_')[1])
        agg_df = df.groupby(group_cols)[band_cols].quantile(percentile / 100.0).reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return agg_df


def _save_csv_with_metadata(
    df: pd.DataFrame,
    output_path: str,
    wavelengths: np.ndarray
):
    """
    Save DataFrame to CSV with wavelength metadata in header.

    Args:
        df: DataFrame to save
        output_path: Output CSV file path
        wavelengths: Array of wavelength values
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, 'w') as f:
        # Write wavelength metadata as comment
        f.write(f"# Wavelengths (nm): {','.join(map(str, wavelengths))}\n")
        f.write(f"# Total pixels: {len(df)}\n")

        # Write DataFrame
        df.to_csv(f, index=False)


def _dataframe_to_blacklibrary(
    df: pd.DataFrame,
    wavelengths: np.ndarray
) -> BlackLibrary:
    """
    Convert DataFrame of spectral signatures to BlackLibrary format.

    Args:
        df: DataFrame with spectral signatures
        wavelengths: Array of wavelength values

    Returns:
        BlackLibrary instance
    """
    # Get band columns
    band_cols = [col for col in df.columns if col.startswith('band_')]

    # Extract spectral data
    spectra = df[band_cols].values

    # Get unique feature names
    if 'feature_name' in df.columns:
        labels = df['feature_name'].unique().tolist()
    else:
        labels = [f"Feature_{i}" for i in range(len(df))]

    # Reshape for BlackLibrary: (n_features, n_samples, n_bands)
    # If aggregated, each row is one spectrum
    # If not aggregated, group by feature
    if 'feature_id' in df.columns and len(labels) < len(df):
        # Not aggregated - group by feature
        grouped_spectra = []
        for label in labels:
            feature_spectra = df[df['feature_name'] == label][band_cols].values
            grouped_spectra.append(feature_spectra)

        # Find max number of samples per feature
        max_samples = max(len(s) for s in grouped_spectra)

        # Pad with NaN to create regular array
        padded_spectra = []
        for feature_spectra in grouped_spectra:
            if len(feature_spectra) < max_samples:
                padding = np.full((max_samples - len(feature_spectra), len(band_cols)), np.nan)
                feature_spectra = np.vstack([feature_spectra, padding])
            padded_spectra.append(feature_spectra)

        data = np.array(padded_spectra)  # Shape: (n_features, n_samples, n_bands)
    else:
        # Aggregated - one spectrum per feature
        data = spectra[np.newaxis, :, :]  # Shape: (1, n_samples, n_bands)
        labels = df['feature_name'].tolist() if 'feature_name' in df.columns else labels

    # Create BlackLibrary
    library = BlackLibrary(data, wav=wavelengths, lab=labels)

    return library