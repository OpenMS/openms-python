"""Pythonic wrapper for :class:`pyopenms.ExperimentalDesign`."""
from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Set
import pandas as pd

import pyopenms as oms

from ._io_utils import ensure_allowed_suffix

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
    def from_file(cls, filepath: Union[str, Path]) -> 'Py_ExperimentalDesign':
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

    def load(self, filepath: Union[str, Path]) -> 'Py_ExperimentalDesign':
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

    def store(self, filepath: Union[str, Path]) -> 'Py_ExperimentalDesign':
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
            'n_samples': self.n_samples,
            'n_ms_files': self.n_ms_files,
            'n_fractions': self.n_fractions,
            'n_fraction_groups': self.n_fraction_groups,
            'n_labels': self.n_labels,
            'is_fractionated': self.is_fractionated,
            'samples': sorted(self.samples),
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
        if summary['samples']:
            print(f"Sample IDs: {', '.join(summary['samples'])}")

    # ==================== Factory methods ====================

    @classmethod
    def from_consensus_map(cls, consensus_map: Union['Py_ConsensusMap', oms.ConsensusMap]) -> 'Py_ExperimentalDesign':
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
        native_map = consensus_map.native if hasattr(consensus_map, 'native') else consensus_map
        design = oms.ExperimentalDesign.fromConsensusMap(native_map)
        return cls(design)

    @classmethod
    def from_feature_map(cls, feature_map: Union['Py_FeatureMap', oms.FeatureMap]) -> 'Py_ExperimentalDesign':
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
        native_map = feature_map.native if hasattr(feature_map, 'native') else feature_map
        design = oms.ExperimentalDesign.fromFeatureMap(native_map)
        return cls(design)

    @classmethod
    def from_identifications(
        cls,
        protein_ids: list
    ) -> 'Py_ExperimentalDesign':
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
