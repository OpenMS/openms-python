"""Pythonic wrapper for :class:`pyopenms.ExperimentalDesign`."""

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Set, TYPE_CHECKING

import pandas as pd
import pyopenms as oms

from ._io_utils import ensure_allowed_suffix

if TYPE_CHECKING:
    from .py_consensusmap import Py_ConsensusMap
    from .py_featuremap import Py_FeatureMap

# Supported file extensions for experimental design
EXPERIMENTAL_DESIGN_EXTENSIONS = {".tsv"}


class Py_ExperimentalDesign:
    """A Pythonic wrapper around :class:`pyopenms.ExperimentalDesign`.

    This class provides convenient methods for loading, storing, and working with
    experimental design files in OpenMS format.

    Example:
        >>> from openms_python import Py_ExperimentalDesign
        >>> design = Py_ExperimentalDesign.from_file("design.tsv")
        >>> print(f"Samples: {design.n_samples}, MS files: {design.n_ms_files}")
    """

    def __init__(self, native_design: Optional[oms.ExperimentalDesign] = None):
        """Initialize with an optional native ExperimentalDesign object.

        Parameters
        ----------
        native_design:
            Optional :class:`pyopenms.ExperimentalDesign` to wrap.
        """
        self._design = native_design if native_design is not None else oms.ExperimentalDesign()

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "Py_ExperimentalDesign":
        """Load an experimental design from a TSV file.

        Parameters
        ----------
        filepath:
            Path to the experimental design TSV file.

        Returns
        -------
        Py_ExperimentalDesign
            A new instance with the loaded design.

        Example:
            >>> design = Py_ExperimentalDesign.from_file("design.tsv")
        """
        instance = cls()
        instance.load(filepath)
        return instance

    def load(self, filepath: Union[str, Path]) -> "Py_ExperimentalDesign":
        """Load an experimental design from disk.

        Parameters
        ----------
        filepath:
            Path to the experimental design TSV file.

        Returns
        -------
        Py_ExperimentalDesign
            Self for method chaining.
        """
        ensure_allowed_suffix(filepath, EXPERIMENTAL_DESIGN_EXTENSIONS, "ExperimentalDesign")
        edf = oms.ExperimentalDesignFile()
        self._design = edf.load(str(filepath), False)
        return self

    def store(self, filepath: Union[str, Path]) -> "Py_ExperimentalDesign":
        """Store the experimental design to disk.

        Note: Storage functionality is not available in the current pyOpenMS API.
        This method is provided for API consistency but will raise NotImplementedError.

        Parameters
        ----------
        filepath:
            Path where the experimental design should be saved.

        Returns
        -------
        Py_ExperimentalDesign
            Self for method chaining.

        Raises
        ------
        NotImplementedError
            Storage is not yet implemented in pyOpenMS.
        """
        ensure_allowed_suffix(filepath, EXPERIMENTAL_DESIGN_EXTENSIONS, "ExperimentalDesign")
        raise NotImplementedError(
            "ExperimentalDesign storage is not yet available in pyOpenMS. "
            "Please use the native pyOpenMS API if this functionality is needed."
        )

    @property
    def native(self) -> oms.ExperimentalDesign:
        """Return the underlying :class:`pyopenms.ExperimentalDesign`."""
        return self._design

    # ==================== Properties ====================

    @property
    def n_samples(self) -> int:
        """Number of samples in the experimental design."""
        return self._design.getNumberOfSamples()

    @property
    def n_ms_files(self) -> int:
        """Number of MS files in the experimental design."""
        return self._design.getNumberOfMSFiles()

    @property
    def n_fractions(self) -> int:
        """Number of fractions in the experimental design."""
        return self._design.getNumberOfFractions()

    @property
    def n_fraction_groups(self) -> int:
        """Number of fraction groups in the experimental design."""
        return self._design.getNumberOfFractionGroups()

    @property
    def n_labels(self) -> int:
        """Number of labels in the experimental design."""
        return self._design.getNumberOfLabels()

    @property
    def is_fractionated(self) -> bool:
        """Whether the experimental design includes fractionation."""
        return self._design.isFractionated()

    @property
    def same_n_ms_files_per_fraction(self) -> bool:
        """Whether all fractions have the same number of MS files."""
        return self._design.sameNrOfMSFilesPerFraction()

    @property
    def samples(self) -> Set[str]:
        """Set of sample identifiers in the design.

        Returns
        -------
        Set[str]
            Set of sample identifiers.
        """
        sample_section = self._design.getSampleSection()
        samples = sample_section.getSamples()
        # Convert bytes to str if needed
        return {s.decode() if isinstance(s, bytes) else str(s) for s in samples}

    # ==================== Summary methods ====================

    def summary(self) -> dict:
        """Get a summary of the experimental design.

        Returns
        -------
        dict
            Dictionary with summary statistics.
        """
        return {
            "n_samples": self.n_samples,
            "n_ms_files": self.n_ms_files,
            "n_fractions": self.n_fractions,
            "n_fraction_groups": self.n_fraction_groups,
            "n_labels": self.n_labels,
            "is_fractionated": self.is_fractionated,
            "samples": sorted(self.samples),
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the experimental design."""
        summary = self.summary()
        print("Experimental Design Summary")
        print("=" * 40)
        print(f"Samples: {summary['n_samples']}")
        print(f"MS Files: {summary['n_ms_files']}")
        print(f"Fractions: {summary['n_fractions']}")
        print(f"Fraction Groups: {summary['n_fraction_groups']}")
        print(f"Labels: {summary['n_labels']}")
        print(f"Fractionated: {summary['is_fractionated']}")
        if summary["samples"]:
            print(f"Sample IDs: {', '.join(summary['samples'])}")

    # ==================== Factory methods ====================

    @classmethod
    def from_consensus_map(
        cls, consensus_map: Union["Py_ConsensusMap", oms.ConsensusMap]
    ) -> "Py_ExperimentalDesign":
        """Create an ExperimentalDesign from a ConsensusMap.

        Parameters
        ----------
        consensus_map:
            A :class:`Py_ConsensusMap` or :class:`pyopenms.ConsensusMap`.

        Returns
        -------
        Py_ExperimentalDesign
            A new instance derived from the consensus map.
        """
        # Handle both Py_ConsensusMap and native ConsensusMap
        native_map = consensus_map.native if hasattr(consensus_map, "native") else consensus_map
        design = oms.ExperimentalDesign.fromConsensusMap(native_map)
        return cls(design)

    @classmethod
    def from_feature_map(
        cls, feature_map: Union["Py_FeatureMap", oms.FeatureMap]
    ) -> "Py_ExperimentalDesign":
        """Create an ExperimentalDesign from a FeatureMap.

        Parameters
        ----------
        feature_map:
            A :class:`Py_FeatureMap` or :class:`pyopenms.FeatureMap`.

        Returns
        -------
        Py_ExperimentalDesign
            A new instance derived from the feature map.
        """
        # Handle both Py_FeatureMap and native FeatureMap
        native_map = feature_map.native if hasattr(feature_map, "native") else feature_map
        design = oms.ExperimentalDesign.fromFeatureMap(native_map)
        return cls(design)

    @classmethod
    def from_identifications(cls, protein_ids: list) -> "Py_ExperimentalDesign":
        """Create an ExperimentalDesign from protein identification data.

        Parameters
        ----------
        protein_ids:
            List of :class:`pyopenms.ProteinIdentification` objects.

        Returns
        -------
        Py_ExperimentalDesign
            A new instance derived from the identifications.
        """
        design = oms.ExperimentalDesign.fromIdentifications(protein_ids)
        return cls(design)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Py_ExperimentalDesign":
        """Create an ExperimentalDesign from a pandas DataFrame.

        Parameters
        ----------
        df:
            DataFrame with columns: Fraction_Group, Fraction, Spectra_Filepath,
            Label, Sample.

        Returns
        -------
        Py_ExperimentalDesign
            A new instance created from the DataFrame.

        Raises
        ------
        ValueError
            If required columns are missing from the DataFrame.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'Fraction_Group': [1, 1, 2, 2],
            ...     'Fraction': [1, 2, 1, 2],
            ...     'Spectra_Filepath': ['f1.mzML', 'f2.mzML', 'f3.mzML', 'f4.mzML'],
            ...     'Label': [1, 1, 1, 1],
            ...     'Sample': [1, 1, 2, 2]
            ... })
            >>> design = Py_ExperimentalDesign.from_dataframe(df)
        """
        import tempfile

        required_columns = {
            "Fraction_Group",
            "Fraction",
            "Spectra_Filepath",
            "Label",
            "Sample",
        }
        missing = required_columns - set(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"DataFrame is missing required columns: {missing_str}. "
                f"Required columns are: {', '.join(sorted(required_columns))}"
            )

        # Write DataFrame to a temporary TSV file and load it
        # This ensures proper sample section setup by the OpenMS loader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            # Write in the expected format
            df.to_csv(f, sep="\t", index=False)
            temp_path = f.name

        try:
            edf = oms.ExperimentalDesignFile()
            design = edf.load(temp_path, False)
            return cls(design)
        finally:
            Path(temp_path).unlink()

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "Py_ExperimentalDesign":
        """Alias for :meth:`from_dataframe` matching :meth:`get_df`.

        Parameters
        ----------
        df:
            DataFrame with experimental design data.

        Returns
        -------
        Py_ExperimentalDesign
            A new instance created from the DataFrame.
        """
        return cls.from_dataframe(df)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the ExperimentalDesign to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Fraction_Group, Fraction, Spectra_Filepath,
            Label, Sample.

        Example:
            >>> design = Py_ExperimentalDesign.from_file("design.tsv")
            >>> df = design.to_dataframe()
        """
        ms_files = self._design.getMSFileSection()

        # Get sample IDs from the sample section
        sample_section = self._design.getSampleSection()
        sample_ids = sorted(sample_section.getSamples())
        # Create index-to-id mapping (0-based index to actual sample ID)
        index_to_sample_id = {}
        for sample_id in sample_ids:
            # Decode if bytes
            if isinstance(sample_id, bytes):
                sample_id_str = sample_id.decode()
            else:
                sample_id_str = str(sample_id)
            # Try to convert to int if possible
            try:
                sample_id_value = int(sample_id_str)
            except ValueError:
                sample_id_value = sample_id_str
            index_to_sample_id[len(index_to_sample_id)] = sample_id_value

        data = {
            "Fraction_Group": [],
            "Fraction": [],
            "Spectra_Filepath": [],
            "Label": [],
            "Sample": [],
        }

        for entry in ms_files:
            data["Fraction_Group"].append(entry.fraction_group)
            data["Fraction"].append(entry.fraction)
            # Decode path if it's bytes
            path = entry.path
            if isinstance(path, bytes):
                path = path.decode()
            data["Spectra_Filepath"].append(path)
            data["Label"].append(entry.label)
            # Map 0-based index back to actual sample ID
            data["Sample"].append(index_to_sample_id.get(entry.sample, entry.sample))

        return pd.DataFrame(data)

    def get_df(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`.

        Returns
        -------
        pd.DataFrame
            DataFrame with experimental design data.
        """
        return self.to_dataframe()

    # ==================== Delegation ====================

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying ExperimentalDesign."""
        return getattr(self._design, name)

    def __repr__(self) -> str:
        """String representation of the ExperimentalDesign."""
        return (
            f"Py_ExperimentalDesign(samples={self.n_samples}, "
            f"ms_files={self.n_ms_files}, fractionated={self.is_fractionated})"
        )
