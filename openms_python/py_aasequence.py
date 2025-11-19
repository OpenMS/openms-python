"""Pythonic wrapper for pyOpenMS AASequence class."""

from __future__ import annotations

from typing import Optional
import pyopenms as oms


class Py_AASequence:
    """
    A Pythonic wrapper around pyOpenMS AASequence.

    This class provides intuitive properties and methods for working with
    amino acid sequences, including common operations like reversing and
    shuffling sequences with optional enzyme constraints.

    Example:
        >>> seq = Py_AASequence.from_string("PEPTIDE")
        >>> print(seq.sequence)
        PEPTIDE
        >>> print(seq.mono_weight)
        799.36...
        >>> reversed_seq = seq.reverse()
        >>> print(reversed_seq.sequence)
        EDITPEP
        >>> shuffled_seq = seq.shuffle(enzyme="Trypsin")
        >>> print(shuffled_seq.sequence)  # Shuffled while preserving cleavage sites
    """

    def __init__(self, native_sequence: Optional[oms.AASequence] = None):
        """
        Initialize Py_AASequence wrapper.

        Args:
            native_sequence: pyOpenMS AASequence object. If None, creates empty sequence.
        """
        self._sequence = native_sequence if native_sequence is not None else oms.AASequence()
        self._decoy_generator = None

    @classmethod
    def from_string(cls, sequence_str: str) -> Py_AASequence:
        """
        Create AASequence from string representation.

        Args:
            sequence_str: String representation of the amino acid sequence.
                         Can include modifications in OpenMS format.

        Returns:
            Py_AASequence: New wrapped sequence.

        Example:
            >>> seq = Py_AASequence.from_string("PEPTIDE")
            >>> seq = Py_AASequence.from_string("PEPTIDEM(Oxidation)")
        """
        return cls(oms.AASequence.fromString(sequence_str))

    # ==================== Pythonic Properties ====================

    @property
    def native(self) -> oms.AASequence:
        """Return the underlying pyOpenMS AASequence."""
        return self._sequence

    @property
    def sequence(self) -> str:
        """Get the sequence as a string."""
        return self._sequence.toString()

    @property
    def unmodified_sequence(self) -> str:
        """Get the sequence without modifications."""
        return self._sequence.toUnmodifiedString()

    @property
    def mono_weight(self) -> float:
        """Get monoisotopic weight."""
        return self._sequence.getMonoWeight()

    @property
    def average_weight(self) -> float:
        """Get average weight."""
        return self._sequence.getAverageWeight()

    @property
    def formula(self) -> str:
        """Get molecular formula."""
        return self._sequence.getFormula().toString()

    @property
    def is_modified(self) -> bool:
        """Check if sequence has any modifications."""
        return self._sequence.isModified()

    @property
    def has_n_terminal_modification(self) -> bool:
        """Check if sequence has N-terminal modification."""
        return self._sequence.hasNTerminalModification()

    @property
    def has_c_terminal_modification(self) -> bool:
        """Check if sequence has C-terminal modification."""
        return self._sequence.hasCTerminalModification()

    # ==================== Decoy Generation ====================

    def _get_decoy_generator(self) -> oms.DecoyGenerator:
        """Get or create DecoyGenerator instance (lazy initialization)."""
        if self._decoy_generator is None:
            self._decoy_generator = oms.DecoyGenerator()
        return self._decoy_generator

    def reverse(self) -> Py_AASequence:
        """
        Reverse the entire amino acid sequence.

        Returns:
            Py_AASequence: New sequence with reversed amino acids.

        Example:
            >>> seq = Py_AASequence.from_string("PEPTIDE")
            >>> reversed_seq = seq.reverse()
            >>> print(reversed_seq.sequence)
            EDITPEP
        """
        dg = self._get_decoy_generator()
        reversed_native = dg.reverseProtein(self._sequence)
        return Py_AASequence(reversed_native)

    def reverse_with_enzyme(self, enzyme: str = "Trypsin") -> Py_AASequence:
        """
        Reverse peptide sequences between enzymatic cleavage sites.

        This is useful for creating decoy sequences that maintain the
        same enzymatic cleavage pattern as the target.

        Args:
            enzyme: Name of the enzyme (e.g., "Trypsin", "Lys-C", "Asp-N").
                   Default is "Trypsin".

        Returns:
            Py_AASequence: New sequence with reversed peptides between cleavage sites.

        Example:
            >>> seq = Py_AASequence.from_string("PEPTIDERK")
            >>> reversed_seq = seq.reverse_with_enzyme("Trypsin")
            >>> # K and R are cleavage sites, so segments are reversed separately
        """
        dg = self._get_decoy_generator()
        reversed_native = dg.reversePeptides(self._sequence, enzyme)
        return Py_AASequence(reversed_native)

    def shuffle(
        self, enzyme: str = "Trypsin", max_attempts: int = 100, seed: Optional[int] = None
    ) -> Py_AASequence:
        """
        Shuffle peptide sequences between enzymatic cleavage sites.

        This creates a decoy sequence by shuffling amino acids within
        peptide segments defined by enzyme cleavage sites, attempting
        to minimize sequence identity with the original.

        Args:
            enzyme: Name of the enzyme (e.g., "Trypsin", "Lys-C", "Asp-N").
                   Default is "Trypsin".
            max_attempts: Maximum number of shuffle attempts to minimize
                         sequence identity. Default is 100.
            seed: Optional random seed for reproducible shuffling.

        Returns:
            Py_AASequence: New shuffled sequence.

        Example:
            >>> seq = Py_AASequence.from_string("PEPTIDERK")
            >>> shuffled_seq = seq.shuffle(enzyme="Trypsin", seed=42)
            >>> # Amino acids are shuffled within enzyme-defined segments
        """
        dg = self._get_decoy_generator()
        if seed is not None:
            dg.setSeed(seed)
        shuffled_native = dg.shufflePeptides(self._sequence, enzyme, max_attempts)
        return Py_AASequence(shuffled_native)

    # ==================== Sequence Operations ====================

    def __len__(self) -> int:
        """Get sequence length."""
        return self._sequence.size()

    def __str__(self) -> str:
        """String representation."""
        return self.sequence

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        seq_str = self.sequence
        if len(seq_str) > 20:
            seq_str = seq_str[:17] + "..."
        return f"Py_AASequence('{seq_str}')"

    def __eq__(self, other: object) -> bool:
        """Check equality based on sequence string."""
        if not isinstance(other, Py_AASequence):
            return False
        return self.sequence == other.sequence

    def __getitem__(self, index: int) -> str:
        """
        Get residue at position.

        Args:
            index: Position in the sequence (0-based).

        Returns:
            str: Single letter amino acid code.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for sequence of length {len(self)}")
        residue = self._sequence.getResidue(index)
        return residue.getOneLetterCode()

    def __iter__(self):
        """Iterate over residues."""
        for i in range(len(self)):
            yield self[i]

    # ==================== Additional Utilities ====================

    def get_mz(self, charge: int) -> float:
        """
        Get m/z value for given charge state.

        Args:
            charge: Charge state (must be > 0).

        Returns:
            float: m/z value.

        Example:
            >>> seq = Py_AASequence.from_string("PEPTIDE")
            >>> mz = seq.get_mz(2)  # doubly charged
        """
        return self._sequence.getMZ(charge)

    def has_substring(self, substring: str) -> bool:
        """
        Check if sequence contains a substring.

        Args:
            substring: Amino acid sequence to search for.

        Returns:
            bool: True if substring is present.
        """
        return self._sequence.hasSubsequence(oms.AASequence.fromString(substring))

    def has_prefix(self, prefix: str) -> bool:
        """
        Check if sequence starts with a prefix.

        Args:
            prefix: Amino acid sequence to check.

        Returns:
            bool: True if sequence starts with prefix.
        """
        return self._sequence.hasPrefix(oms.AASequence.fromString(prefix))

    def has_suffix(self, suffix: str) -> bool:
        """
        Check if sequence ends with a suffix.

        Args:
            suffix: Amino acid sequence to check.

        Returns:
            bool: True if sequence ends with suffix.
        """
        return self._sequence.hasSuffix(oms.AASequence.fromString(suffix))

