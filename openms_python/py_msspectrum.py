"""
Pythonic wrapper for pyOpenMS Spectrum classes.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import pyopenms as oms
from pyopenms import Constants

from ._meta_mapping import MetaInfoMappingMixin


class Py_MSSpectrum(MetaInfoMappingMixin):
    """
    A Pythonic wrapper around pyOpenMS MSSpectrum.

    This class provides intuitive properties and methods for working with
    mass spectra, hiding the verbose C++ API underneath.

    Example:
        >>> spec = Spectrum(native_spectrum)
        >>> print(f"RT: {spec.retention_time:.2f} seconds")
        >>> print(f"MS Level: {spec.ms_level}")
        >>> print(f"Number of peaks: {len(spec)}")
        >>> if spec.is_ms1:
        ...     print("This is an MS1 spectrum")
        >>> peaks_df = spec.to_dataframe()
    """

    def __init__(self, native_spectrum: oms.MSSpectrum):
        """
        Initialize Spectrum wrapper.

        Args:
            native_spectrum: pyOpenMS MSSpectrum object
        """
        self._spectrum = native_spectrum

    # ==================== Meta-info support ====================

    def _meta_object(self) -> oms.MetaInfoInterface:
        return self._spectrum

    # ==================== Pythonic Properties ====================

    @property
    def retention_time(self) -> float:
        """Get retention time in seconds."""
        return self._spectrum.getRT()

    @retention_time.setter
    def retention_time(self, value: float):
        """Set retention time in seconds."""
        self._spectrum.setRT(value)

    @property
    def ms_level(self) -> int:
        """Get MS level (1 for MS1, 2 for MS2, etc.)."""
        return self._spectrum.getMSLevel()

    @ms_level.setter
    def ms_level(self, value: int):
        """Set MS level."""
        self._spectrum.setMSLevel(value)

    @property
    def is_ms1(self) -> bool:
        """Check if this is an MS1 spectrum."""
        return self.ms_level == 1

    @property
    def is_ms2(self) -> bool:
        """Check if this is an MS2 spectrum."""
        return self.ms_level == 2

    @property
    def precursor_mz(self) -> Optional[float]:
        """Get precursor m/z for MS2+ spectra, None for MS1."""
        if self.ms_level < 2:
            return None
        precursors = self._spectrum.getPrecursors()
        if precursors:
            return precursors[0].getMZ()
        return None

    @property
    def precursor_charge(self) -> Optional[int]:
        """Get precursor charge for MS2+ spectra, None for MS1."""
        if self.ms_level < 2:
            return None
        precursors = self._spectrum.getPrecursors()
        if precursors:
            return precursors[0].getCharge()
        return None

    @property
    def precursor_mass(self) -> Optional[float]:
        """Return the neutral precursor mass when available."""

        if self.ms_level < 2:
            return None
        precursors = self._spectrum.getPrecursors()
        if not precursors:
            return None
        precursor = precursors[0]
        uncharged = float(precursor.getUnchargedMass())
        if uncharged > 0:
            return uncharged
        charge = precursor.getCharge()
        mz = precursor.getMZ()
        if charge <= 0 or mz <= 0:
            return None
        proton = Constants.PROTON_MASS_U
        return float((mz - proton) * charge)

    @property
    def native_id(self) -> str:
        """Get native ID of the spectrum."""
        return self._spectrum.getNativeID()

    @property
    def scan_number(self) -> int:
        """Get scan number (extracted from native ID or -1 if not available)."""
        try:
            # Try to extract scan number from native ID
            native_id = self.native_id
            if "scan=" in native_id:
                return int(native_id.split("scan=")[1].split()[0])
        except (ValueError, IndexError):
            pass
        return -1

    @property
    def total_ion_current(self) -> float:
        """Get total ion current (sum of all peak intensities)."""
        _, intensities = self.peaks
        return float(np.sum(intensities))

    @property
    def base_peak_mz(self) -> Optional[float]:
        """Get m/z of the base peak (most intense peak)."""
        if len(self) == 0:
            return None
        mz, intensities = self.peaks
        return float(mz[np.argmax(intensities)])

    @property
    def base_peak_intensity(self) -> Optional[float]:
        """Get intensity of the base peak."""
        if len(self) == 0:
            return None
        _, intensities = self.peaks
        return float(np.max(intensities))

    @property
    def peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get peaks as NumPy arrays.

        Returns:
            Tuple of (mz_array, intensity_array)
        """
        mz, intensity = self._spectrum.get_peaks()
        return np.array(mz), np.array(intensity)

    @peaks.setter
    def peaks(self, values: Tuple[np.ndarray, np.ndarray]):
        """
        Set peaks from NumPy arrays.

        Args:
            values: Tuple of (mz_array, intensity_array)
        """
        mz, intensity = values
        self._spectrum.set_peaks((mz.tolist(), intensity.tolist()))

    @property
    def mz(self) -> np.ndarray:
        """Get m/z values as a NumPy array."""
        mz, _ = self.peaks
        return mz

    @property
    def intensity(self) -> np.ndarray:
        """Get intensity values as a NumPy array."""
        _, intensity = self.peaks
        return intensity

    @property
    def float_data_arrays(self) -> list:
        """
        Get float data arrays attached to this spectrum.

        Returns:
            List of pyOpenMS FloatDataArray objects

        Example:
            >>> arrays = spec.float_data_arrays
            >>> for arr in arrays:
            ...     print(f"Array: {arr.getName()}, size: {arr.size()}")
        """
        return self._spectrum.getFloatDataArrays()

    @float_data_arrays.setter
    def float_data_arrays(self, arrays: list):
        """
        Set float data arrays on this spectrum.

        Args:
            arrays: List of pyOpenMS FloatDataArray objects
        """
        self._spectrum.setFloatDataArrays(arrays)

    @property
    def ion_mobility(self) -> Optional[np.ndarray]:
        """
        Get ion mobility values as a NumPy array if available.

        Returns:
            NumPy array of ion mobility values, or None if not available

        Example:
            >>> if spec.ion_mobility is not None:
            ...     print(f"Ion mobility: {spec.ion_mobility}")
        """
        float_arrays = self.float_data_arrays
        for arr in float_arrays:
            name = arr.getName().lower()
            if "ion" in name and "mobility" in name or name == "ion_mobility":
                return np.array([arr[i] for i in range(arr.size())])
        return None

    @ion_mobility.setter
    def ion_mobility(self, values: np.ndarray):
        """
        Set ion mobility values for this spectrum.

        Creates or updates a FloatDataArray named 'ion_mobility' with the provided values.
        The array should have the same length as the number of peaks.

        Args:
            values: NumPy array of ion mobility values

        Example:
            >>> spec.ion_mobility = np.array([1.5, 2.3, 3.1])
        """
        if len(values) != len(self):
            raise ValueError(
                f"Ion mobility array length ({len(values)}) must match "
                f"number of peaks ({len(self)})"
            )

        # Get existing arrays
        float_arrays = self.float_data_arrays

        # Find or create ion mobility array
        ion_mobility_array = None
        ion_mobility_index = -1
        for i, arr in enumerate(float_arrays):
            name = arr.getName().lower()
            if "ion" in name and "mobility" in name or name == "ion_mobility":
                ion_mobility_array = arr
                ion_mobility_index = i
                break

        if ion_mobility_array is None:
            # Create new array
            ion_mobility_array = oms.FloatDataArray()
            ion_mobility_array.setName("ion_mobility")
            float_arrays.append(ion_mobility_array)

        # Clear and set new values
        ion_mobility_array.clear()
        for val in values:
            ion_mobility_array.push_back(float(val))

        # Update the arrays on the spectrum
        if ion_mobility_index >= 0:
            float_arrays[ion_mobility_index] = ion_mobility_array

        self.float_data_arrays = float_arrays

    @property
    def drift_time(self) -> float:
        """
        Get the drift time of this spectrum.

        Returns:
            Drift time value, or -1.0 if not set
        """
        return self._spectrum.getDriftTime()

    @drift_time.setter
    def drift_time(self, value: float):
        """Set the drift time of this spectrum."""
        self._spectrum.setDriftTime(value)

    # ==================== Magic Methods ====================

    def __len__(self) -> int:
        """Return number of peaks in the spectrum."""
        return self._spectrum.size()

    def __repr__(self) -> str:
        """Return string representation."""
        ms_info = f"MS{self.ms_level}"
        if self.precursor_mz is not None:
            ms_info += f" (precursor: {self.precursor_mz:.4f})"
        return (
            f"Spectrum(rt={self.retention_time:.2f}s, {ms_info}, "
            f"peaks={len(self)}, TIC={self.total_ion_current:.2e})"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return self.__repr__()

    def __iter__(self):
        """Allow dict(self) and list(self) conversions."""
        yield "mz", self.mz.tolist()
        yield "intens", self.intensity.tolist()

    # ==================== Conversion Methods ====================

    def to_numpy(self) -> np.ndarray:
        """
        Convert spectrum peaks to NumPy arrays.

        Returns:
            Tuple of (mz_array, intensity_array)

        Example:
            >>> mz, intensity = spec.to_numpy()
        """
        return np.array(self.peaks)

    def to_dataframe(self, include_float_arrays: bool = True) -> pd.DataFrame:
        """
        Convert spectrum peaks to pandas DataFrame.

        Args:
            include_float_arrays: If True, include float data arrays as columns

        Returns:
            DataFrame with columns: mz, intensity, and any float data arrays

        Example:
            >>> df = spec.to_dataframe()
            >>> df.head()
                      mz   intensity  ion_mobility
            0  100.0500     1250.5          1.5
            1  200.1234     5678.2          2.3
            ...
        """
        mz, intensity = self.peaks
        data = {"mz": mz, "intensity": intensity}

        # Add float data arrays if requested
        if include_float_arrays:
            float_arrays = self.float_data_arrays
            for arr in float_arrays:
                name = arr.getName()
                if name and arr.size() == len(mz):
                    data[name] = np.array([arr[i] for i in range(arr.size())])

        return pd.DataFrame(data)

    @classmethod
    def from_numpy(mz: np.ndarray, intensity: np.ndarray) -> "Py_MSSpectrum":
        """
        Create spectrum from NumPy arrays.

        Args:
            mz: Array of m/z values
            intensity: Array of intensity values
        """
        spec = oms.MSSpectrum()
        spec.set_peaks((mz.tolist(), intensity.tolist()))
        return Py_MSSpectrum(spec)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **metadata) -> "Py_MSSpectrum":
        """
        Create spectrum from pandas DataFrame.

        Args:
            df: DataFrame with 'mz' and 'intensity' columns, and optionally other float data arrays
            **metadata: Optional metadata (retention_time, ms_level, etc.)

        Returns:
            Spectrum object

        Example:
            >>> df = pd.DataFrame({
            ...     'mz': [100, 200],
            ...     'intensity': [50, 100],
            ...     'ion_mobility': [1.5, 2.3]
            ... })
            >>> spec = Spectrum.from_dataframe(df, retention_time=60.5, ms_level=1)
        """
        spec = oms.MSSpectrum()
        spec.set_peaks((df["mz"].values.tolist(), df["intensity"].values.tolist()))

        # Set metadata
        if "retention_time" in metadata:
            spec.setRT(metadata["retention_time"])
        if "ms_level" in metadata:
            spec.setMSLevel(metadata["ms_level"])
        if "native_id" in metadata:
            spec.setNativeID(metadata["native_id"])

        # Add float data arrays from DataFrame columns (excluding mz and intensity)
        float_arrays = []
        for col in df.columns:
            if col not in ["mz", "intensity"]:
                fda = oms.FloatDataArray()
                fda.setName(col)
                for val in df[col].values:
                    fda.push_back(float(val))
                float_arrays.append(fda)

        if float_arrays:
            spec.setFloatDataArrays(float_arrays)

        return cls(spec)

    # ==================== Data Manipulation ====================

    def filter_by_mz(self, min_mz: float, max_mz: float) -> "Py_MSSpectrum":
        """
        Filter peaks by m/z range.

        Args:
            min_mz: Minimum m/z value
            max_mz: Maximum m/z value

        Returns:
            New Spectrum with filtered peaks
        """
        mz, intensity = self.peaks
        mask = (mz >= min_mz) & (mz <= max_mz)

        new_spec = oms.MSSpectrum()
        new_spec.set_peaks((mz[mask].tolist(), intensity[mask].tolist()))
        new_spec.setRT(self.retention_time)
        new_spec.setMSLevel(self.ms_level)
        new_spec.setNativeID(self.native_id)

        return Py_MSSpectrum(new_spec)

    def filter_by_intensity(self, min_intensity: float) -> "Py_MSSpectrum":
        """
        Filter peaks by minimum intensity.

        Args:
            min_intensity: Minimum intensity threshold

        Returns:
            New Spectrum with filtered peaks
        """
        mz, intensity = self.peaks
        mask = intensity >= min_intensity

        new_spec = oms.MSSpectrum()
        new_spec.set_peaks((mz[mask].tolist(), intensity[mask].tolist()))
        new_spec.setRT(self.retention_time)
        new_spec.setMSLevel(self.ms_level)
        new_spec.setNativeID(self.native_id)

        return Py_MSSpectrum(new_spec)

    def top_n_peaks(self, n: int) -> "Py_MSSpectrum":
        """
        Keep only the top N most intense peaks.

        Args:
            n: Number of peaks to keep

        Returns:
            New Spectrum with top N peaks
        """
        mz, intensity = self.peaks
        if len(mz) <= n:
            return self

        # Get indices of top N peaks
        top_indices = np.argsort(intensity)[-n:]
        top_indices = np.sort(top_indices)  # Keep m/z order

        new_spec = oms.MSSpectrum()
        new_spec.set_peaks((mz[top_indices].tolist(), intensity[top_indices].tolist()))
        new_spec.setRT(self.retention_time)
        new_spec.setMSLevel(self.ms_level)
        new_spec.setNativeID(self.native_id)

        return Py_MSSpectrum(new_spec)

    def normalize_to_tic(self) -> "Py_MSSpectrum":
        """Scale intensities so their sum equals one."""

        mz, intensity = self.peaks
        total = float(np.sum(intensity))
        if total <= 0:
            return self

        normalized_spec = oms.MSSpectrum(self._spectrum)
        normalized_spec.set_peaks((mz.tolist(), (intensity / total).tolist()))
        return Py_MSSpectrum(normalized_spec)

    def normalize_intensity(self, max_value: float = 100.0) -> "Py_MSSpectrum":
        """
        Normalize peak intensities to a maximum value.

        Args:
            max_value: Target maximum intensity (default: 100.0)

        Returns:
            New Spectrum with normalized intensities
        """
        mz, intensity = self.peaks
        if len(intensity) == 0 or np.max(intensity) == 0:
            return self

        normalized = intensity * (max_value / np.max(intensity))

        new_spec = oms.MSSpectrum()
        new_spec.set_peaks((mz.tolist(), normalized.tolist()))
        new_spec.setRT(self.retention_time)
        new_spec.setMSLevel(self.ms_level)
        new_spec.setNativeID(self.native_id)

        return Py_MSSpectrum(new_spec)

    # ==================== Access to Native Object ====================

    @property
    def native(self) -> oms.MSSpectrum:
        """
        Get the underlying pyOpenMS MSSpectrum object.

        Use this when you need to access pyOpenMS-specific methods
        not wrapped by this class.
        """
        return self._spectrum
