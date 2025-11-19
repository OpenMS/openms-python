"""
Pythonic wrapper for pyOpenMS MSChromatogram for mobilogram representation.

A mobilogram is a chromatogram in the ion mobility dimension, representing
intensity vs. drift time for a specific m/z value.

Note: OpenMS C++ has a native Mobilogram class that may not yet be wrapped
in pyopenms. This wrapper uses MSChromatogram as the underlying representation
for mobilogram data.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import pyopenms as oms

from ._meta_mapping import MetaInfoMappingMixin


class Py_Mobilogram(MetaInfoMappingMixin):
    """
    A Pythonic wrapper around pyOpenMS MSChromatogram for mobilograms.

    A mobilogram represents the ion mobility dimension for a specific m/z,
    showing intensity vs. drift time (or other ion mobility values).

    Note: OpenMS C++ has a native Mobilogram class that may not yet be wrapped
    in pyopenms. This wrapper uses MSChromatogram as the underlying representation
    for mobilogram data.

    Example:
        >>> mob = Py_Mobilogram(native_chromatogram)
        >>> print(f"m/z: {mob.mz:.4f}")
        >>> print(f"Number of points: {len(mob)}")
        >>> drift_times, intensities = mob.peaks
        >>> df = mob.to_dataframe()
    """

    def __init__(self, native_chromatogram: oms.MSChromatogram):
        """
        Initialize Mobilogram wrapper.

        Args:
            native_chromatogram: pyOpenMS MSChromatogram object
        """
        self._chromatogram = native_chromatogram

    # ==================== Meta-info support ====================

    def _meta_object(self) -> oms.MetaInfoInterface:
        return self._chromatogram

    # ==================== Pythonic Properties ====================

    @property
    def name(self) -> str:
        """Get the name of this mobilogram."""
        return self._chromatogram.getName()

    @name.setter
    def name(self, value: str):
        """Set the name of this mobilogram."""
        self._chromatogram.setName(value)

    @property
    def mz(self) -> Optional[float]:
        """
        Get the m/z value this mobilogram represents.

        Returns from metadata if available, None otherwise.
        """
        if self._chromatogram.metaValueExists("mz"):
            return float(self._chromatogram.getMetaValue("mz"))
        return None

    @mz.setter
    def mz(self, value: float):
        """Set the m/z value this mobilogram represents."""
        self._chromatogram.setMetaValue("mz", value)

    @property
    def drift_time(self) -> np.ndarray:
        """
        Get drift time values as a NumPy array.

        Returns:
            NumPy array of drift time values
        """
        rt, _ = self.peaks
        return rt

    @property
    def intensity(self) -> np.ndarray:
        """
        Get intensity values as a NumPy array.

        Returns:
            NumPy array of intensity values
        """
        _, intensity = self.peaks
        return intensity

    @property
    def peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mobilogram data as NumPy arrays.

        Returns:
            Tuple of (drift_time_array, intensity_array)
        """
        rt, intensity = self._chromatogram.get_peaks()
        return np.array(rt), np.array(intensity)

    @peaks.setter
    def peaks(self, values: Tuple[np.ndarray, np.ndarray]):
        """
        Set mobilogram data from NumPy arrays.

        Args:
            values: Tuple of (drift_time_array, intensity_array)
        """
        drift_time, intensity = values
        # Clear existing peaks
        self._chromatogram.clear(False)
        # Add new peaks
        for dt, i in zip(drift_time, intensity):
            peak = oms.ChromatogramPeak()
            peak.setRT(float(dt))
            peak.setIntensity(float(i))
            self._chromatogram.push_back(peak)

    @property
    def total_ion_current(self) -> float:
        """Get total ion current (sum of all intensities)."""
        return float(np.sum(self.intensity))

    @property
    def base_peak_drift_time(self) -> Optional[float]:
        """Get drift time of the base peak (most intense point)."""
        if len(self) == 0:
            return None
        drift_time, intensities = self.peaks
        return float(drift_time[np.argmax(intensities)])

    @property
    def base_peak_intensity(self) -> Optional[float]:
        """Get intensity of the base peak."""
        if len(self) == 0:
            return None
        return float(np.max(self.intensity))

    # ==================== Magic Methods ====================

    def __len__(self) -> int:
        """Return number of points in the mobilogram."""
        return self._chromatogram.size()

    def __repr__(self) -> str:
        """Return string representation."""
        mz_str = f"m/z={self.mz:.4f}" if self.mz is not None else "m/z=unset"
        return f"Mobilogram({mz_str}, points={len(self)}, " f"TIC={self.total_ion_current:.2e})"

    def __str__(self) -> str:
        """Return human-readable string."""
        return self.__repr__()

    # ==================== Conversion Methods ====================

    def to_numpy(self) -> np.ndarray:
        """
        Convert mobilogram to NumPy arrays.

        Returns:
            Tuple of (drift_time_array, intensity_array)
        """
        return np.array(self.peaks)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert mobilogram to pandas DataFrame.

        Returns:
            DataFrame with columns: drift_time, intensity

        Example:
            >>> df = mob.to_dataframe()
            >>> df.head()
                drift_time  intensity
            0         1.5      100.0
            1         2.0      150.0
            ...
        """
        drift_time, intensity = self.peaks
        data = {"drift_time": drift_time, "intensity": intensity}

        # Add m/z as a column if available
        if self.mz is not None:
            data["mz"] = self.mz

        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **metadata) -> "Py_Mobilogram":
        """
        Create mobilogram from pandas DataFrame.

        Args:
            df: DataFrame with 'drift_time' and 'intensity' columns
            **metadata: Optional metadata (name, mz, etc.)

        Returns:
            Mobilogram object

        Example:
            >>> df = pd.DataFrame({
            ...     'drift_time': [1.5, 2.0, 2.5],
            ...     'intensity': [100, 150, 120]
            ... })
            >>> mob = Py_Mobilogram.from_dataframe(df, mz=500.0, name="mobilogram_500")
        """
        chrom = oms.MSChromatogram()

        # Add peaks
        for dt, intensity in zip(df["drift_time"].values, df["intensity"].values):
            peak = oms.ChromatogramPeak()
            peak.setRT(float(dt))
            peak.setIntensity(float(intensity))
            chrom.push_back(peak)

        # Set metadata
        if "name" in metadata:
            chrom.setName(metadata["name"])

        # Get m/z from metadata or DataFrame
        mz_value = metadata.get("mz")
        if mz_value is None and "mz" in df.columns:
            # Extract m/z from DataFrame (take first value if it's a column)
            mz_value = float(df["mz"].iloc[0])

        if mz_value is not None:
            chrom.setMetaValue("mz", float(mz_value))

        # Mark as mobilogram type
        chrom.setMetaValue("chromatogram_type", "mobilogram")

        return cls(chrom)

    @classmethod
    def from_arrays(
        cls,
        drift_time: np.ndarray,
        intensity: np.ndarray,
        mz: Optional[float] = None,
        name: Optional[str] = None,
    ) -> "Py_Mobilogram":
        """
        Create mobilogram from NumPy arrays.

        Args:
            drift_time: Array of drift time values
            intensity: Array of intensity values
            mz: Optional m/z value this mobilogram represents
            name: Optional name for the mobilogram

        Returns:
            Mobilogram object

        Example:
            >>> mob = Py_Mobilogram.from_arrays(
            ...     np.array([1.5, 2.0, 2.5]),
            ...     np.array([100, 150, 120]),
            ...     mz=500.0
            ... )
        """
        chrom = oms.MSChromatogram()

        # Add peaks
        for dt, i in zip(drift_time, intensity):
            peak = oms.ChromatogramPeak()
            peak.setRT(float(dt))
            peak.setIntensity(float(i))
            chrom.push_back(peak)

        # Set metadata
        if name:
            chrom.setName(name)
        if mz is not None:
            chrom.setMetaValue("mz", float(mz))

        # Mark as mobilogram type
        chrom.setMetaValue("chromatogram_type", "mobilogram")

        return cls(chrom)

    # ==================== Access to Native Object ====================

    @property
    def native(self) -> oms.MSChromatogram:
        """
        Get the underlying pyOpenMS MSChromatogram object.

        Use this when you need to access pyOpenMS-specific methods
        not wrapped by this class.
        """
        return self._chromatogram
