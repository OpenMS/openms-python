"""
Tests for Py_MSChromatogram wrapper.
"""

import pytest
import numpy as np
import pandas as pd
import pyopenms as oms

from openms_python import Py_MSChromatogram


def test_py_chromatogram_properties():
    """Test basic properties of Py_MSChromatogram."""
    # Create a native chromatogram
    native_chrom = oms.MSChromatogram()
    product = native_chrom.getProduct()
    product.setMZ(445.12)
    native_chrom.setProduct(product)
    native_chrom.setName("XIC 445.12")
    native_chrom.setNativeID("chromatogram=1")
    native_chrom.set_peaks(([10.0, 20.0, 30.0], [100.0, 200.0, 150.0]))
    
    # Wrap it
    chrom = Py_MSChromatogram(native_chrom)
    
    # Test properties
    assert chrom.mz == pytest.approx(445.12)
    assert chrom.name == "XIC 445.12"
    assert chrom.native_id == "chromatogram=1"
    assert len(chrom) == 3
    
    # Test RT range
    assert chrom.min_rt == pytest.approx(10.0)
    assert chrom.max_rt == pytest.approx(30.0)
    assert chrom.rt_range == (pytest.approx(10.0), pytest.approx(30.0))
    
    # Test intensity range
    assert chrom.max_intensity == pytest.approx(200.0)
    assert chrom.min_intensity == pytest.approx(100.0)
    assert chrom.total_ion_current == pytest.approx(450.0)


def test_py_chromatogram_data_access():
    """Test data access methods."""
    native_chrom = oms.MSChromatogram()
    native_chrom.set_peaks(([10.0, 20.0, 30.0], [100.0, 200.0, 150.0]))
    
    chrom = Py_MSChromatogram(native_chrom)
    
    # Test data property
    rt, intensity = chrom.data
    assert isinstance(rt, np.ndarray)
    assert isinstance(intensity, np.ndarray)
    assert len(rt) == 3
    assert len(intensity) == 3
    np.testing.assert_array_almost_equal(rt, [10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(intensity, [100.0, 200.0, 150.0])
    
    # Test individual accessors
    np.testing.assert_array_almost_equal(chrom.rt, [10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(chrom.intensity, [100.0, 200.0, 150.0])


def test_py_chromatogram_setters():
    """Test property setters."""
    chrom = Py_MSChromatogram(oms.MSChromatogram())
    
    # Test setters
    chrom.mz = 500.25
    assert chrom.mz == pytest.approx(500.25)
    
    chrom.name = "Test Chromatogram"
    assert chrom.name == "Test Chromatogram"
    
    chrom.native_id = "chrom=test"
    assert chrom.native_id == "chrom=test"
    
    # Test data setter
    chrom.data = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
    rt, intensity = chrom.data
    np.testing.assert_array_almost_equal(rt, [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(intensity, [10.0, 20.0, 30.0])


def test_py_chromatogram_from_dataframe():
    """Test creating chromatogram from DataFrame."""
    df = pd.DataFrame({
        'rt': [10.0, 20.0, 30.0],
        'intensity': [100.0, 200.0, 150.0]
    })
    
    chrom = Py_MSChromatogram.from_dataframe(
        df,
        mz=445.12,
        name="Test XIC",
        native_id="chrom=1"
    )
    
    assert len(chrom) == 3
    assert chrom.mz == pytest.approx(445.12)
    assert chrom.name == "Test XIC"
    assert chrom.native_id == "chrom=1"
    
    rt, intensity = chrom.data
    np.testing.assert_array_almost_equal(rt, [10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(intensity, [100.0, 200.0, 150.0])


def test_py_chromatogram_to_dataframe():
    """Test converting chromatogram to DataFrame."""
    native_chrom = oms.MSChromatogram()
    native_chrom.set_peaks(([10.0, 20.0, 30.0], [100.0, 200.0, 150.0]))
    
    chrom = Py_MSChromatogram(native_chrom)
    df = chrom.to_dataframe()
    
    assert isinstance(df, pd.DataFrame)
    assert 'rt' in df.columns
    assert 'intensity' in df.columns
    assert len(df) == 3
    
    np.testing.assert_array_almost_equal(df['rt'].values, [10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(df['intensity'].values, [100.0, 200.0, 150.0])


def test_py_chromatogram_dataframe_roundtrip():
    """Test DataFrame round-trip conversion."""
    original_df = pd.DataFrame({
        'rt': [10.5, 20.3, 30.1],
        'intensity': [123.4, 234.5, 178.9]
    })
    
    chrom = Py_MSChromatogram.from_dataframe(original_df, mz=500.0)
    result_df = chrom.to_dataframe()
    
    # Check values with tolerance for float32/float64 conversion
    np.testing.assert_array_almost_equal(result_df['rt'].values, original_df['rt'].values, decimal=5)
    np.testing.assert_array_almost_equal(result_df['intensity'].values, original_df['intensity'].values, decimal=5)


def test_py_chromatogram_from_numpy():
    """Test creating chromatogram from NumPy arrays."""
    rt = np.array([10.0, 20.0, 30.0])
    intensity = np.array([100.0, 200.0, 150.0])
    
    chrom = Py_MSChromatogram.from_numpy(
        rt, intensity,
        mz=445.12,
        name="NumPy Chromatogram"
    )
    
    assert len(chrom) == 3
    assert chrom.mz == pytest.approx(445.12)
    assert chrom.name == "NumPy Chromatogram"
    
    rt_out, intensity_out = chrom.data
    np.testing.assert_array_almost_equal(rt_out, rt)
    np.testing.assert_array_almost_equal(intensity_out, intensity)


def test_py_chromatogram_filter_by_rt():
    """Test filtering by retention time."""
    native_chrom = oms.MSChromatogram()
    native_chrom.set_peaks(([10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 150.0, 120.0]))
    product = native_chrom.getProduct()
    product.setMZ(445.12)
    native_chrom.setProduct(product)
    
    chrom = Py_MSChromatogram(native_chrom)
    filtered = chrom.filter_by_rt(15.0, 35.0)
    
    assert len(filtered) == 2
    rt, intensity = filtered.data
    np.testing.assert_array_almost_equal(rt, [20.0, 30.0])
    np.testing.assert_array_almost_equal(intensity, [200.0, 150.0])
    
    # Ensure metadata is preserved
    assert filtered.mz == pytest.approx(445.12)


def test_py_chromatogram_filter_by_intensity():
    """Test filtering by intensity."""
    native_chrom = oms.MSChromatogram()
    native_chrom.set_peaks(([10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 150.0, 80.0]))
    native_chrom.setName("Test")
    
    chrom = Py_MSChromatogram(native_chrom)
    filtered = chrom.filter_by_intensity(100.0)
    
    assert len(filtered) == 3
    rt, intensity = filtered.data
    np.testing.assert_array_almost_equal(rt, [10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(intensity, [100.0, 200.0, 150.0])
    
    # Ensure metadata is preserved
    assert filtered.name == "Test"


def test_py_chromatogram_normalize_intensity():
    """Test intensity normalization."""
    native_chrom = oms.MSChromatogram()
    native_chrom.set_peaks(([10.0, 20.0, 30.0], [100.0, 200.0, 150.0]))
    
    chrom = Py_MSChromatogram(native_chrom)
    normalized = chrom.normalize_intensity(max_value=1000.0)
    
    assert len(normalized) == 3
    rt, intensity = normalized.data
    np.testing.assert_array_almost_equal(rt, [10.0, 20.0, 30.0])
    assert intensity.max() == pytest.approx(1000.0)
    assert intensity[0] == pytest.approx(500.0)  # 100/200 * 1000
    assert intensity[2] == pytest.approx(750.0)  # 150/200 * 1000


def test_py_chromatogram_normalize_to_tic():
    """Test normalization to TIC."""
    native_chrom = oms.MSChromatogram()
    native_chrom.set_peaks(([10.0, 20.0, 30.0], [100.0, 200.0, 150.0]))
    
    chrom = Py_MSChromatogram(native_chrom)
    normalized = chrom.normalize_to_tic()
    
    assert len(normalized) == 3
    assert normalized.total_ion_current == pytest.approx(1.0)


def test_py_chromatogram_empty():
    """Test behavior with empty chromatogram."""
    chrom = Py_MSChromatogram(oms.MSChromatogram())
    
    assert len(chrom) == 0
    assert chrom.total_ion_current == 0.0
    assert chrom.rt_range == (0.0, 0.0)
    assert chrom.min_rt == 0.0
    assert chrom.max_rt == 0.0
    
    df = chrom.to_dataframe()
    assert len(df) == 0


def test_py_chromatogram_repr():
    """Test string representation."""
    native_chrom = oms.MSChromatogram()
    product = native_chrom.getProduct()
    product.setMZ(445.12)
    native_chrom.setProduct(product)
    native_chrom.setName("XIC")
    native_chrom.set_peaks(([10.0, 20.0], [100.0, 200.0]))
    
    chrom = Py_MSChromatogram(native_chrom)
    repr_str = repr(chrom)
    
    assert "445.12" in repr_str
    assert "XIC" in repr_str
    assert "points=2" in repr_str


def test_py_chromatogram_meta_mapping():
    """Test dictionary-style meta data access."""
    chrom = Py_MSChromatogram(oms.MSChromatogram())
    
    # Set and get meta values
    chrom["custom_label"] = "test_sample"
    chrom["scan_rate"] = 5.0
    
    assert chrom["custom_label"] == "test_sample"
    assert chrom["scan_rate"] == 5.0
    
    # Test 'in' operator
    assert "custom_label" in chrom
    assert "nonexistent" not in chrom
    
    # Test get with default
    assert chrom.get("custom_label") == "test_sample"
    assert chrom.get("missing", "default") == "default"


def test_py_chromatogram_native_access():
    """Test access to native pyOpenMS object."""
    native_chrom = oms.MSChromatogram()
    chrom = Py_MSChromatogram(native_chrom)
    
    assert chrom.native is native_chrom
    assert isinstance(chrom.native, oms.MSChromatogram)
