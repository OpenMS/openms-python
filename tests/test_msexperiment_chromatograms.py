"""
Tests for MSExperiment chromatogram integration.
"""

import pytest
import numpy as np
import pandas as pd
import pyopenms as oms

from openms_python import Py_MSExperiment, Py_MSChromatogram


def test_msexperiment_chromatogram_count():
    """Test chromatogram count properties."""
    exp = Py_MSExperiment()
    
    # Initially empty
    assert exp.nr_chromatograms == 0
    assert exp.chromatogram_count == 0
    
    # Add a chromatogram
    chrom = Py_MSChromatogram(oms.MSChromatogram())
    exp.add_chromatogram(chrom)
    
    assert exp.nr_chromatograms == 1
    assert exp.chromatogram_count == 1


def test_msexperiment_add_chromatogram():
    """Test adding chromatograms to experiment."""
    exp = Py_MSExperiment()
    
    # Create chromatogram from DataFrame
    df = pd.DataFrame({
        'rt': [10.0, 20.0, 30.0],
        'intensity': [100.0, 200.0, 150.0]
    })
    chrom = Py_MSChromatogram.from_dataframe(df, mz=445.12, name="XIC 445.12")
    
    # Add it
    exp.add_chromatogram(chrom)
    
    assert exp.nr_chromatograms == 1
    
    # Add another
    chrom2 = Py_MSChromatogram.from_dataframe(df, mz=500.0, name="XIC 500.0")
    exp.add_chromatogram(chrom2)
    
    assert exp.nr_chromatograms == 2


def test_msexperiment_get_chromatogram():
    """Test getting chromatograms by index."""
    exp = Py_MSExperiment()
    
    # Add chromatograms
    df1 = pd.DataFrame({'rt': [10.0, 20.0], 'intensity': [100.0, 200.0]})
    chrom1 = Py_MSChromatogram.from_dataframe(df1, mz=445.12, name="First")
    exp.add_chromatogram(chrom1)
    
    df2 = pd.DataFrame({'rt': [30.0, 40.0], 'intensity': [150.0, 250.0]})
    chrom2 = Py_MSChromatogram.from_dataframe(df2, mz=500.0, name="Second")
    exp.add_chromatogram(chrom2)
    
    # Get them back
    retrieved1 = exp.get_chromatogram(0)
    assert retrieved1.mz == pytest.approx(445.12)
    assert retrieved1.name == "First"
    assert len(retrieved1) == 2
    
    retrieved2 = exp.get_chromatogram(1)
    assert retrieved2.mz == pytest.approx(500.0)
    assert retrieved2.name == "Second"
    assert len(retrieved2) == 2


def test_msexperiment_chromatograms_iteration():
    """Test iterating over chromatograms."""
    exp = Py_MSExperiment()
    
    # Add multiple chromatograms
    mz_values = [445.12, 500.0, 550.25]
    for mz in mz_values:
        df = pd.DataFrame({'rt': [10.0, 20.0, 30.0], 'intensity': [100.0, 200.0, 150.0]})
        chrom = Py_MSChromatogram.from_dataframe(df, mz=mz)
        exp.add_chromatogram(chrom)
    
    # Iterate and check
    retrieved_mzs = []
    for chrom in exp.chromatograms():
        assert isinstance(chrom, Py_MSChromatogram)
        assert len(chrom) == 3
        retrieved_mzs.append(chrom.mz)
    
    assert len(retrieved_mzs) == 3
    np.testing.assert_array_almost_equal(retrieved_mzs, mz_values)


def test_msexperiment_repr_with_chromatograms():
    """Test string representation includes chromatogram count."""
    exp = Py_MSExperiment()
    
    # Add a spectrum
    spec = oms.MSSpectrum()
    spec.setRT(10.0)
    spec.setMSLevel(1)
    spec.set_peaks(([100.0, 200.0], [50.0, 100.0]))
    exp._experiment.addSpectrum(spec)
    
    # Initially no chromatograms
    repr_str = repr(exp)
    assert "chromatograms" not in repr_str
    
    # Add a chromatogram
    df = pd.DataFrame({'rt': [10.0, 20.0], 'intensity': [100.0, 200.0]})
    chrom = Py_MSChromatogram.from_dataframe(df, mz=445.12)
    exp.add_chromatogram(chrom)
    
    repr_str = repr(exp)
    assert "chromatograms=1" in repr_str


def test_msexperiment_add_native_chromatogram():
    """Test adding native pyOpenMS chromatogram."""
    exp = Py_MSExperiment()
    
    # Create native chromatogram
    native_chrom = oms.MSChromatogram()
    native_chrom.set_peaks(([10.0, 20.0, 30.0], [100.0, 200.0, 150.0]))
    product = native_chrom.getProduct()
    product.setMZ(445.12)
    native_chrom.setProduct(product)
    
    # Add it directly
    exp.add_chromatogram(native_chrom)
    
    assert exp.nr_chromatograms == 1
    
    # Retrieve and verify
    retrieved = exp.get_chromatogram(0)
    assert retrieved.mz == pytest.approx(445.12)
    assert len(retrieved) == 3


def test_msexperiment_empty_chromatograms():
    """Test experiment with no chromatograms."""
    exp = Py_MSExperiment()
    
    # Should not raise error
    assert exp.nr_chromatograms == 0
    
    # Should return empty iterator
    chroms = list(exp.chromatograms())
    assert len(chroms) == 0
