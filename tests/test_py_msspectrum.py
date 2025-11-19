import numpy as np
import pandas as pd
import pytest

oms = pytest.importorskip("pyopenms")

from openms_python.py_msspectrum import Py_MSSpectrum


def create_native_spectrum():
    spec = oms.MSSpectrum()
    spec.setRT(12.3)
    spec.setMSLevel(2)
    spec.setNativeID("controllerType=0 controllerNumber=1 scan=15")
    spec.set_peaks((
        [100.5, 150.2, 300.4, 320.5],
        [200.0, 500.0, 50.0, 1000.0],
    ))
    precursor = oms.Precursor()
    precursor.setMZ(543.21)
    precursor.setCharge(3)
    spec.setPrecursors([precursor])
    return spec


def test_py_msspectrum_properties_and_peak_operations():
    wrapper = Py_MSSpectrum(create_native_spectrum())

    assert pytest.approx(wrapper.retention_time) == 12.3
    wrapper.retention_time = 14.6
    assert pytest.approx(wrapper.retention_time) == 14.6

    assert wrapper.ms_level == 2
    wrapper.ms_level = 1
    assert wrapper.is_ms1
    assert not wrapper.is_ms2
    wrapper.ms_level = 2  # restore original

    assert wrapper.precursor_mz == pytest.approx(543.21)
    assert wrapper.precursor_charge == 3
    assert wrapper.scan_number == 15
    assert len(wrapper) == 4

    mz, intensity = wrapper.peaks
    assert np.allclose(mz, np.array([100.5, 150.2, 300.4, 320.5]))
    assert np.allclose(intensity, np.array([200.0, 500.0, 50.0, 1000.0]))

    assert wrapper.total_ion_current == pytest.approx(1750.0)
    assert wrapper.base_peak_mz == pytest.approx(320.5)
    assert wrapper.base_peak_intensity == pytest.approx(1000.0)

    filtered = wrapper.filter_by_mz(120.0, 310.0)
    assert np.all(filtered.mz >= 120.0) and np.all(filtered.mz <= 310.0)
    assert len(filtered) == 2

    filtered_intensity = wrapper.filter_by_intensity(400.0)
    assert np.all(filtered_intensity.intensity >= 400.0)
    assert len(filtered_intensity) == 2

    top_one = wrapper.top_n_peaks(1)
    assert len(top_one) == 1
    assert top_one.base_peak_mz == pytest.approx(320.5)

    normalized = wrapper.normalize_intensity(max_value=100.0)
    assert normalized.base_peak_intensity == pytest.approx(100.0)

    new_mz = np.array([50.0, 75.0])
    new_intensity = np.array([10.0, 20.0])
    wrapper.peaks = (new_mz, new_intensity)
    assert np.allclose(wrapper.mz, new_mz)
    assert np.allclose(wrapper.intensity, new_intensity)


def test_py_msspectrum_dataframe_helpers_round_trip():
    df = pd.DataFrame({"mz": [100.0, 200.0], "intensity": [10.0, 20.0]})

    wrapper = Py_MSSpectrum.from_dataframe(df, retention_time=5.5, ms_level=1, native_id="scan=1")

    recreated_df = wrapper.to_dataframe()
    assert recreated_df.shape == (2, 2)
    assert recreated_df["mz"].tolist() == [100.0, 200.0]
    assert recreated_df["intensity"].tolist() == [10.0, 20.0]
    assert wrapper.retention_time == pytest.approx(5.5)
    assert wrapper.ms_level == 1
    assert wrapper.native_id == "scan=1"


def test_py_msspectrum_float_data_arrays():
    """Test float data array support."""
    spec = oms.MSSpectrum()
    spec.set_peaks(([100.0, 200.0, 300.0], [50.0, 100.0, 75.0]))
    
    wrapper = Py_MSSpectrum(spec)
    
    # Initially no float arrays
    assert len(wrapper.float_data_arrays) == 0
    assert wrapper.ion_mobility is None
    
    # Add a float data array
    fda = oms.FloatDataArray()
    fda.setName("test_array")
    fda.push_back(1.0)
    fda.push_back(2.0)
    fda.push_back(3.0)
    
    float_arrays = wrapper.float_data_arrays
    float_arrays.append(fda)
    wrapper.float_data_arrays = float_arrays
    
    # Verify it was added
    assert len(wrapper.float_data_arrays) == 1
    assert wrapper.float_data_arrays[0].getName() == "test_array"
    assert wrapper.float_data_arrays[0].size() == 3


def test_py_msspectrum_ion_mobility():
    """Test ion mobility convenience properties."""
    spec = oms.MSSpectrum()
    spec.set_peaks(([100.0, 200.0, 300.0], [50.0, 100.0, 75.0]))
    
    wrapper = Py_MSSpectrum(spec)
    
    # Initially no ion mobility
    assert wrapper.ion_mobility is None
    
    # Set ion mobility
    im_values = np.array([1.5, 2.3, 3.1])
    wrapper.ion_mobility = im_values
    
    # Verify it was set
    assert wrapper.ion_mobility is not None
    assert len(wrapper.ion_mobility) == 3
    assert np.allclose(wrapper.ion_mobility, im_values, rtol=1e-5)
    
    # Verify float array was created
    assert len(wrapper.float_data_arrays) == 1
    assert wrapper.float_data_arrays[0].getName() == "ion_mobility"


def test_py_msspectrum_ion_mobility_wrong_length():
    """Test that setting ion mobility with wrong length raises error."""
    spec = oms.MSSpectrum()
    spec.set_peaks(([100.0, 200.0, 300.0], [50.0, 100.0, 75.0]))
    
    wrapper = Py_MSSpectrum(spec)
    
    # Try to set ion mobility with wrong length
    with pytest.raises(ValueError, match="Ion mobility array length"):
        wrapper.ion_mobility = np.array([1.5, 2.3])  # Only 2 values, should be 3


def test_py_msspectrum_drift_time():
    """Test drift time property."""
    spec = oms.MSSpectrum()
    spec.set_peaks(([100.0, 200.0], [50.0, 100.0]))
    
    wrapper = Py_MSSpectrum(spec)
    
    # Initially -1.0 (not set)
    assert wrapper.drift_time == -1.0
    
    # Set drift time
    wrapper.drift_time = 5.5
    assert wrapper.drift_time == pytest.approx(5.5)


def test_py_msspectrum_dataframe_with_float_arrays():
    """Test DataFrame conversion with float data arrays."""
    df = pd.DataFrame({
        "mz": [100.0, 200.0, 300.0],
        "intensity": [50.0, 100.0, 75.0],
        "ion_mobility": [1.5, 2.3, 3.1]
    })
    
    # Create spectrum from DataFrame
    wrapper = Py_MSSpectrum.from_dataframe(df, retention_time=12.3, ms_level=1)
    
    # Verify peaks were set
    assert len(wrapper) == 3
    assert np.allclose(wrapper.mz, [100.0, 200.0, 300.0])
    assert np.allclose(wrapper.intensity, [50.0, 100.0, 75.0])
    
    # Verify ion mobility was set
    assert wrapper.ion_mobility is not None
    assert np.allclose(wrapper.ion_mobility, [1.5, 2.3, 3.1], rtol=1e-5)
    
    # Convert back to DataFrame
    df_back = wrapper.to_dataframe(include_float_arrays=True)
    assert "mz" in df_back.columns
    assert "intensity" in df_back.columns
    assert "ion_mobility" in df_back.columns
    assert np.allclose(df_back["ion_mobility"], [1.5, 2.3, 3.1], rtol=1e-5)
    
    # Test without float arrays
    df_no_arrays = wrapper.to_dataframe(include_float_arrays=False)
    assert "mz" in df_no_arrays.columns
    assert "intensity" in df_no_arrays.columns
    assert "ion_mobility" not in df_no_arrays.columns


def test_py_msspectrum_dataframe_with_multiple_float_arrays():
    """Test DataFrame conversion with multiple float data arrays."""
    df = pd.DataFrame({
        "mz": [100.0, 200.0],
        "intensity": [50.0, 100.0],
        "ion_mobility": [1.5, 2.3],
        "custom_array": [10.0, 20.0]
    })
    
    # Create spectrum from DataFrame
    wrapper = Py_MSSpectrum.from_dataframe(df)
    
    # Verify both float arrays were created
    assert len(wrapper.float_data_arrays) == 2
    
    # Convert back to DataFrame
    df_back = wrapper.to_dataframe(include_float_arrays=True)
    assert "ion_mobility" in df_back.columns
    assert "custom_array" in df_back.columns
    assert np.allclose(df_back["ion_mobility"], [1.5, 2.3], rtol=1e-5)
    assert np.allclose(df_back["custom_array"], [10.0, 20.0], rtol=1e-5)

