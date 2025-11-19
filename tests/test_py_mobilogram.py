import numpy as np
import pandas as pd
import pytest

oms = pytest.importorskip("pyopenms")

from openms_python.py_mobilogram import Py_Mobilogram


def create_native_chromatogram():
    """Create a native pyOpenMS MSChromatogram for testing."""
    chrom = oms.MSChromatogram()
    chrom.setName("test_mobilogram")
    chrom.setMetaValue("mz", 500.0)
    chrom.setMetaValue("chromatogram_type", "mobilogram")
    
    # Add some peaks
    for dt, intensity in [(1.5, 100.0), (2.0, 150.0), (2.5, 120.0), (3.0, 80.0)]:
        peak = oms.ChromatogramPeak()
        peak.setRT(dt)
        peak.setIntensity(intensity)
        chrom.push_back(peak)
    
    return chrom


def test_py_mobilogram_properties():
    """Test basic mobilogram properties."""
    wrapper = Py_Mobilogram(create_native_chromatogram())
    
    assert wrapper.name == "test_mobilogram"
    assert wrapper.mz == pytest.approx(500.0)
    assert len(wrapper) == 4
    
    # Test drift time and intensity
    drift_time = wrapper.drift_time
    intensity = wrapper.intensity
    
    assert len(drift_time) == 4
    assert len(intensity) == 4
    assert np.allclose(drift_time, [1.5, 2.0, 2.5, 3.0])
    assert np.allclose(intensity, [100.0, 150.0, 120.0, 80.0])
    
    # Test peaks property
    dt, i = wrapper.peaks
    assert np.allclose(dt, drift_time)
    assert np.allclose(i, intensity)


def test_py_mobilogram_setters():
    """Test mobilogram setters."""
    chrom = oms.MSChromatogram()
    wrapper = Py_Mobilogram(chrom)
    
    # Set name
    wrapper.name = "my_mobilogram"
    assert wrapper.name == "my_mobilogram"
    
    # Set m/z
    wrapper.mz = 600.0
    assert wrapper.mz == pytest.approx(600.0)
    
    # Set peaks
    new_dt = np.array([1.0, 2.0, 3.0])
    new_intensity = np.array([50.0, 100.0, 75.0])
    wrapper.peaks = (new_dt, new_intensity)
    
    assert len(wrapper) == 3
    assert np.allclose(wrapper.drift_time, new_dt)
    assert np.allclose(wrapper.intensity, new_intensity)


def test_py_mobilogram_statistics():
    """Test mobilogram statistical properties."""
    wrapper = Py_Mobilogram(create_native_chromatogram())
    
    assert wrapper.total_ion_current == pytest.approx(450.0)
    assert wrapper.base_peak_drift_time == pytest.approx(2.0)
    assert wrapper.base_peak_intensity == pytest.approx(150.0)


def test_py_mobilogram_to_dataframe():
    """Test conversion to pandas DataFrame."""
    wrapper = Py_Mobilogram(create_native_chromatogram())
    
    df = wrapper.to_dataframe()
    
    assert "drift_time" in df.columns
    assert "intensity" in df.columns
    assert "mz" in df.columns
    
    assert len(df) == 4
    assert np.allclose(df["drift_time"], [1.5, 2.0, 2.5, 3.0])
    assert np.allclose(df["intensity"], [100.0, 150.0, 120.0, 80.0])
    assert np.all(df["mz"] == 500.0)


def test_py_mobilogram_from_dataframe():
    """Test creation from pandas DataFrame."""
    df = pd.DataFrame({
        "drift_time": [1.5, 2.0, 2.5],
        "intensity": [100.0, 150.0, 120.0]
    })
    
    wrapper = Py_Mobilogram.from_dataframe(df, mz=500.0, name="test_mob")
    
    assert wrapper.name == "test_mob"
    assert wrapper.mz == pytest.approx(500.0)
    assert len(wrapper) == 3
    assert np.allclose(wrapper.drift_time, [1.5, 2.0, 2.5])
    assert np.allclose(wrapper.intensity, [100.0, 150.0, 120.0])


def test_py_mobilogram_from_arrays():
    """Test creation from NumPy arrays."""
    drift_time = np.array([1.0, 2.0, 3.0, 4.0])
    intensity = np.array([50.0, 100.0, 75.0, 25.0])
    
    wrapper = Py_Mobilogram.from_arrays(drift_time, intensity, mz=600.0, name="array_mob")
    
    assert wrapper.name == "array_mob"
    assert wrapper.mz == pytest.approx(600.0)
    assert len(wrapper) == 4
    assert np.allclose(wrapper.drift_time, drift_time)
    assert np.allclose(wrapper.intensity, intensity)


def test_py_mobilogram_roundtrip():
    """Test DataFrame round-trip conversion."""
    # Create from arrays
    drift_time = np.array([1.5, 2.0, 2.5, 3.0])
    intensity = np.array([100.0, 150.0, 120.0, 80.0])
    
    mob1 = Py_Mobilogram.from_arrays(drift_time, intensity, mz=500.0)
    
    # Convert to DataFrame
    df = mob1.to_dataframe()
    
    # Create from DataFrame
    mob2 = Py_Mobilogram.from_dataframe(df, name="roundtrip_mob")
    
    # Verify they match
    assert len(mob1) == len(mob2)
    assert np.allclose(mob1.drift_time, mob2.drift_time)
    assert np.allclose(mob1.intensity, mob2.intensity)
    assert mob1.mz == mob2.mz


def test_py_mobilogram_repr():
    """Test string representation."""
    wrapper = Py_Mobilogram(create_native_chromatogram())
    
    repr_str = repr(wrapper)
    assert "Mobilogram" in repr_str
    assert "m/z=500" in repr_str
    assert "points=4" in repr_str
    
    # Test mobilogram without m/z
    chrom = oms.MSChromatogram()
    peak = oms.ChromatogramPeak()
    peak.setRT(1.0)
    peak.setIntensity(100.0)
    chrom.push_back(peak)
    
    wrapper_no_mz = Py_Mobilogram(chrom)
    repr_str_no_mz = repr(wrapper_no_mz)
    assert "m/z=unset" in repr_str_no_mz


def test_py_mobilogram_empty():
    """Test empty mobilogram."""
    chrom = oms.MSChromatogram()
    wrapper = Py_Mobilogram(chrom)
    
    assert len(wrapper) == 0
    assert wrapper.base_peak_drift_time is None
    assert wrapper.base_peak_intensity is None
    assert wrapper.total_ion_current == 0.0


def test_py_mobilogram_native_access():
    """Test access to native pyOpenMS object."""
    chrom = create_native_chromatogram()
    wrapper = Py_Mobilogram(chrom)
    
    native = wrapper.native
    assert isinstance(native, oms.MSChromatogram)
    assert native.size() == 4
