"""
Pythonic wrapper for pyOpenMS MSChromatogram class.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import pyopenms as oms

from ._meta_mapping import MetaInfoMappingMixin


class Py_MSChromatogram(MetaInfoMappingMixin):
    """
    A Pythonic wrapper around pyOpenMS MSChromatogram.
    
    This class provides intuitive properties and methods for working with
    chromatograms, hiding the verbose C++ API underneath.
    
    Example:
        >>> chrom = Py_MSChromatogram(native_chromatogram)
        >>> print(f"MZ: {chrom.mz:.4f}")
        >>> print(f"Name: {chrom.name}")
        >>> print(f"Number of data points: {len(chrom)}")
        >>> peaks_df = chrom.to_dataframe()
    """
    
    def __init__(self, native_chromatogram: oms.MSChromatogram):
        """
        Initialize Chromatogram wrapper.
        
        Args:
            native_chromatogram: pyOpenMS MSChromatogram object
        """
        self._chromatogram = native_chromatogram

    # ==================== Meta-info support ====================

    def _meta_object(self) -> oms.MetaInfoInterface:
        return self._chromatogram
    
    # ==================== Pythonic Properties ====================
    
    @property
    def mz(self) -> float:
        """Get m/z value for this chromatogram."""
        return self._chromatogram.getMZ()
    
    @mz.setter
    def mz(self, value: float):
        """Set m/z value for this chromatogram."""
        product = self._chromatogram.getProduct()
        product.setMZ(value)
        self._chromatogram.setProduct(product)
    
    @property
    def name(self) -> str:
        """Get name of the chromatogram."""
        return self._chromatogram.getName()
    
    @name.setter
    def name(self, value: str):
        """Set name of the chromatogram."""
        self._chromatogram.setName(value)
    
    @property
    def native_id(self) -> str:
        """Get native ID of the chromatogram."""
        return self._chromatogram.getNativeID()
    
    @native_id.setter
    def native_id(self, value: str):
        """Set native ID of the chromatogram."""
        self._chromatogram.setNativeID(value)
    
    @property
    def chromatogram_type(self) -> int:
        """Get chromatogram type."""
        return self._chromatogram.getChromatogramType()
    
    @chromatogram_type.setter
    def chromatogram_type(self, value: int):
        """Set chromatogram type."""
        self._chromatogram.setChromatogramType(value)
    
    @property
    def rt_range(self) -> Tuple[float, float]:
        """Get retention time range (min, max)."""
        if len(self) == 0:
            return (0.0, 0.0)
        rt, _ = self.data
        return (float(np.min(rt)), float(np.max(rt)))
    
    @property
    def min_rt(self) -> float:
        """Get minimum retention time."""
        if len(self) == 0:
            return 0.0
        return float(self._chromatogram.getMinRT())
    
    @property
    def max_rt(self) -> float:
        """Get maximum retention time."""
        if len(self) == 0:
            return 0.0
        return float(self._chromatogram.getMaxRT())
    
    @property
    def max_intensity(self) -> float:
        """Get maximum intensity."""
        if len(self) == 0:
            return 0.0
        return float(self._chromatogram.getMaxIntensity())
    
    @property
    def min_intensity(self) -> float:
        """Get minimum intensity."""
        if len(self) == 0:
            return 0.0
        return float(self._chromatogram.getMinIntensity())
    
    @property
    def total_ion_current(self) -> float:
        """Get total ion current (sum of all intensities)."""
        _, intensities = self.data
        return float(np.sum(intensities))
    
    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get chromatogram data as NumPy arrays.
        
        Returns:
            Tuple of (rt_array, intensity_array)
        """
        rt, intensity = self._chromatogram.get_peaks()
        return np.array(rt), np.array(intensity)
    
    @data.setter
    def data(self, values: Tuple[np.ndarray, np.ndarray]):
        """
        Set chromatogram data from NumPy arrays.
        
        Args:
            values: Tuple of (rt_array, intensity_array)
        """
        rt, intensity = values
        self._chromatogram.set_peaks((rt.tolist(), intensity.tolist()))

    @property
    def rt(self) -> np.ndarray:
        """Get retention time values as a NumPy array."""
        rt, _ = self.data
        return rt

    @property
    def intensity(self) -> np.ndarray:
        """Get intensity values as a NumPy array."""
        _, intensity = self.data
        return intensity
    
    # ==================== Magic Methods ====================
    
    def __len__(self) -> int:
        """Return number of data points in the chromatogram."""
        return self._chromatogram.size()
    
    def __repr__(self) -> str:
        """Return string representation."""
        mz_str = f"m/z={self.mz:.4f}" if self.mz > 0 else "m/z=N/A"
        name_str = f", name='{self.name}'" if self.name else ""
        return (
            f"Chromatogram({mz_str}{name_str}, "
            f"points={len(self)}, TIC={self.total_ion_current:.2e})"
        )
    
    def __str__(self) -> str:
        """Return human-readable string."""
        return self.__repr__()

    def __iter__(self):
        """Allow dict(self) and list(self) conversions."""
        yield "rt", self.rt.tolist()
        yield "intensity", self.intensity.tolist()
    
    # ==================== Conversion Methods ====================

    def to_numpy(self) -> np.ndarray:
        """
        Convert chromatogram data to NumPy arrays.
        
        Returns:
            Tuple of (rt_array, intensity_array)
            
        Example:
            >>> rt, intensity = chrom.to_numpy()
        """
        return np.array(self.data)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert chromatogram data to pandas DataFrame.
        
        Returns:
            DataFrame with columns: rt, intensity
            
        Example:
            >>> df = chrom.to_dataframe()
            >>> df.head()
                      rt   intensity
            0    10.050     1250.5
            1    10.100     5678.2
            ...
        """
        rt, intensity = self.data
        return pd.DataFrame({
            'rt': rt,
            'intensity': intensity
        })
    
    @classmethod
    def from_numpy(cls, rt: np.ndarray, intensity: np.ndarray, **metadata) -> 'Py_MSChromatogram':
        """
        Create chromatogram from NumPy arrays.
        
        Args:
            rt: Array of retention time values
            intensity: Array of intensity values
            **metadata: Optional metadata (mz, name, native_id, etc.)
            
        Returns:
            Py_MSChromatogram object
        """
        chrom = oms.MSChromatogram()
        chrom.set_peaks((rt.tolist(), intensity.tolist()))
        
        # Set metadata
        if 'mz' in metadata:
            product = chrom.getProduct()
            product.setMZ(metadata['mz'])
            chrom.setProduct(product)
        if 'name' in metadata:
            chrom.setName(metadata['name'])
        if 'native_id' in metadata:
            chrom.setNativeID(metadata['native_id'])
        if 'chromatogram_type' in metadata:
            chrom.setChromatogramType(metadata['chromatogram_type'])
            
        return cls(chrom)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **metadata) -> 'Py_MSChromatogram':
        """
        Create chromatogram from pandas DataFrame.
        
        Args:
            df: DataFrame with 'rt' and 'intensity' columns
            **metadata: Optional metadata (mz, name, native_id, etc.)
            
        Returns:
            Py_MSChromatogram object
            
        Example:
            >>> df = pd.DataFrame({'rt': [10.0, 20.0], 'intensity': [50, 100]})
            >>> chrom = Py_MSChromatogram.from_dataframe(df, mz=445.12, name='XIC')
        """
        chrom = oms.MSChromatogram()
        chrom.set_peaks((df['rt'].values.tolist(), df['intensity'].values.tolist()))
        
        # Set metadata
        if 'mz' in metadata:
            product = chrom.getProduct()
            product.setMZ(metadata['mz'])
            chrom.setProduct(product)
        if 'name' in metadata:
            chrom.setName(metadata['name'])
        if 'native_id' in metadata:
            chrom.setNativeID(metadata['native_id'])
        if 'chromatogram_type' in metadata:
            chrom.setChromatogramType(metadata['chromatogram_type'])
            
        return cls(chrom)
    
    # ==================== Data Manipulation ====================

    def filter_by_rt(self, min_rt: float, max_rt: float) -> 'Py_MSChromatogram':
        """
        Filter data points by retention time range.
        
        Args:
            min_rt: Minimum retention time
            max_rt: Maximum retention time
            
        Returns:
            New Py_MSChromatogram with filtered data points
        """
        rt, intensity = self.data
        mask = (rt >= min_rt) & (rt <= max_rt)
        
        new_chrom = oms.MSChromatogram()
        new_chrom.set_peaks((rt[mask].tolist(), intensity[mask].tolist()))
        
        # Copy metadata
        product = self._chromatogram.getProduct()
        new_chrom.setProduct(product)
        new_chrom.setName(self.name)
        new_chrom.setNativeID(self.native_id)
        new_chrom.setChromatogramType(self.chromatogram_type)
        
        return Py_MSChromatogram(new_chrom)
    
    def filter_by_intensity(self, min_intensity: float) -> 'Py_MSChromatogram':
        """
        Filter data points by minimum intensity.
        
        Args:
            min_intensity: Minimum intensity threshold
            
        Returns:
            New Py_MSChromatogram with filtered data points
        """
        rt, intensity = self.data
        mask = intensity >= min_intensity
        
        new_chrom = oms.MSChromatogram()
        new_chrom.set_peaks((rt[mask].tolist(), intensity[mask].tolist()))
        
        # Copy metadata
        product = self._chromatogram.getProduct()
        new_chrom.setProduct(product)
        new_chrom.setName(self.name)
        new_chrom.setNativeID(self.native_id)
        new_chrom.setChromatogramType(self.chromatogram_type)
        
        return Py_MSChromatogram(new_chrom)

    def normalize_to_tic(self) -> 'Py_MSChromatogram':
        """
        Scale intensities so their sum equals one.
        
        Returns:
            New Py_MSChromatogram with normalized intensities
        """
        rt, intensity = self.data
        total = float(np.sum(intensity))
        if total <= 0:
            return self

        normalized_chrom = oms.MSChromatogram(self._chromatogram)
        normalized_chrom.set_peaks((rt.tolist(), (intensity / total).tolist()))
        return Py_MSChromatogram(normalized_chrom)

    def normalize_intensity(self, max_value: float = 100.0) -> 'Py_MSChromatogram':
        """
        Normalize intensities to a maximum value.
        
        Args:
            max_value: Target maximum intensity (default: 100.0)
            
        Returns:
            New Py_MSChromatogram with normalized intensities
        """
        rt, intensity = self.data
        if len(intensity) == 0 or np.max(intensity) == 0:
            return self
        
        normalized = intensity * (max_value / np.max(intensity))
        
        new_chrom = oms.MSChromatogram()
        new_chrom.set_peaks((rt.tolist(), normalized.tolist()))
        
        # Copy metadata
        product = self._chromatogram.getProduct()
        new_chrom.setProduct(product)
        new_chrom.setName(self.name)
        new_chrom.setNativeID(self.native_id)
        new_chrom.setChromatogramType(self.chromatogram_type)

        return Py_MSChromatogram(new_chrom)

    # ==================== Access to Native Object ====================
    
    @property
    def native(self) -> oms.MSChromatogram:
        """
        Get the underlying pyOpenMS MSChromatogram object.
        
        Use this when you need to access pyOpenMS-specific methods
        not wrapped by this class.
        """
        return self._chromatogram
