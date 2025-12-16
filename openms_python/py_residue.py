"""Pythonic wrapper for pyOpenMS Residue class."""

from __future__ import annotations

from typing import Optional, Set, List
import pyopenms as oms


class Py_Residue:
    """
    A Pythonic, immutable wrapper around pyOpenMS Residue.

    This class provides intuitive properties and methods for working with
    amino acid residues, including access to modifications, formulas,
    weights, and physicochemical properties.

    Example:
        >>> # Get a residue from the database
        >>> res = Py_Residue.from_one_letter_code("A")
        >>> print(res.name)
        Alanine
        >>> print(res.mono_weight)
        89.047...
        >>> print(res.formula)
        C3H7NO2
        >>> # Check if modified
        >>> print(res.is_modified)
        False
    """

    def __init__(self, native_residue: Optional[oms.Residue] = None):
        """
        Initialize Py_Residue wrapper.

        Args:
            native_residue: pyOpenMS Residue object. If None, creates empty residue.
        """
        self._residue = native_residue if native_residue is not None else oms.Residue()

    @classmethod
    def from_native(cls, native_residue: oms.Residue) -> Py_Residue:
        """
        Create Py_Residue from native pyOpenMS Residue.

        Args:
            native_residue: Native pyOpenMS Residue object.

        Returns:
            Py_Residue: New wrapped object.
        """
        return cls(native_residue)

    @classmethod
    def from_string(cls, code: str) -> Py_Residue:
        """
        Get residue from ResidueDB by any valid identifier.

        Intelligently infers the input format and retrieves the residue accordingly.
        Supports:
        - One-letter codes: "A", "R", "N", etc.
        - Three-letter codes: "Ala", "Arg", "Asn", etc.
        - Full names: "Alanine", "Arginine", "Asparagine", etc.

        Args:
            code: One-letter code, three-letter code, or full name.

        Returns:
            Py_Residue: Wrapped residue from database.

        Raises:
            ValueError: If the code format cannot be recognized or residue not found.

        Example:
            >>> res = Py_Residue.from_string("A")
            >>> res = Py_Residue.from_string("Ala")
            >>> res = Py_Residue.from_string("Alanine")
        """
        if not code or not isinstance(code, str):
            raise ValueError(f"Invalid code: {code}")

        code = code.strip()
        db = oms.ResidueDB()

        # Try direct lookup (works for all formats)
        try:
            residue = db.getResidue(code)
            return cls(residue)
        except Exception:
            raise ValueError(f"Residue '{code}' not found in ResidueDB.")

    # ==================== Core Properties ====================

    @property
    def native(self) -> oms.Residue:
        """Return the underlying pyOpenMS Residue."""
        return self._residue

    @property
    def name(self) -> str:
        """Get the full name of the residue."""
        return self._residue.getName()

    @property
    def one_letter_code(self) -> str:
        """Get the one-letter code."""
        return self._residue.getOneLetterCode()

    @property
    def three_letter_code(self) -> str:
        """Get the three-letter code."""
        return self._residue.getThreeLetterCode()

    @property
    def synonyms(self) -> Set[str]:
        """Get synonyms for this residue."""
        return self._residue.getSynonyms()

    # ==================== Weight and Formula ====================

    @property
    def mono_weight(self) -> float:
        """
        Get monoisotopic weight.

        Args can be passed to get weight for different residue types
        (Full, Internal, NTerminal, CTerminal, etc.).
        """
        return self._residue.getMonoWeight()

    @property
    def average_weight(self) -> float:
        """Get average weight."""
        return self._residue.getAverageWeight()

    @property
    def formula(self) -> str:
        """
        Get empirical formula.

        Returns:
            str: Formula string (e.g., 'C3H7NO2').
        """
        return self._residue.getFormula().toString()

    def get_mono_weight(
        self, residue_type: oms.Residue.ResidueType = oms.Residue.ResidueType.Full
    ) -> float:
        """
        Get monoisotopic weight for specific residue type.

        Args:
            residue_type: Type of residue (Full, Internal, NTerminal, CTerminal,
                         AIon, BIon, CIon, XIon, YIon, ZIon, etc.).

        Returns:
            float: Monoisotopic weight.

        Example:
            >>> res = Py_Residue.from_one_letter_code("A")
            >>> full_weight = res.get_mono_weight(oms.Residue.ResidueType.Full)
            >>> internal_weight = res.get_mono_weight(oms.Residue.ResidueType.Internal)
        """
        return self._residue.getMonoWeight(residue_type)

    def get_average_weight(
        self, residue_type: oms.Residue.ResidueType = oms.Residue.ResidueType.Full
    ) -> float:
        """
        Get average weight for specific residue type.

        Args:
            residue_type: Type of residue.

        Returns:
            float: Average weight.
        """
        return self._residue.getAverageWeight(residue_type)

    def get_formula(
        self, residue_type: oms.Residue.ResidueType = oms.Residue.ResidueType.Full
    ) -> str:
        """
        Get empirical formula for specific residue type.

        Args:
            residue_type: Type of residue.

        Returns:
            str: Formula string.
        """
        return self._residue.getFormula(residue_type).toString()

    # ==================== Modifications ====================

    @property
    def is_modified(self) -> bool:
        """Check if residue has a modification."""
        return self._residue.isModified()

    @property
    def modification(self) -> Optional[oms.ResidueModification]:
        """
        Get the modification object.

        Returns:
            Optional[oms.ResidueModification]: Modification or None if not modified.
        """
        return self._residue.getModification()

    @property
    def modification_name(self) -> str:
        """
        Get the modification name.

        Returns:
            str: Modification name or empty string if not modified.
        """
        return self._residue.getModificationName()

    def set_modification(self, mod_name: str) -> None:
        """
        Set modification by name.

        Note: This modifies the underlying residue object.

        Args:
            mod_name: Name of modification (must exist in ModificationsDB).

        Example:
            >>> res = Py_Residue.from_one_letter_code("M")
            >>> res.set_modification("Oxidation")
        """
        self._residue.setModification(mod_name)

    def set_modification_by_diff_mass(self, diff_mono_mass: float) -> None:
        """
        Set modification by mass difference.

        Searches ModificationsDB for matching modification. If not found,
        creates a new user-defined modification.

        Args:
            diff_mono_mass: Monoisotopic mass difference.

        Example:
            >>> res = Py_Residue.from_one_letter_code("S")
            >>> res.set_modification_by_diff_mass(79.966331)  # Phosphorylation
        """
        self._residue.setModificationByDiffMonoMass(diff_mono_mass)

    # ==================== Neutral Losses ====================

    @property
    def has_neutral_loss(self) -> bool:
        """Check if residue has neutral losses."""
        return self._residue.hasNeutralLoss()

    @property
    def has_n_term_neutral_losses(self) -> bool:
        """Check if residue has N-terminal neutral losses."""
        return self._residue.hasNTermNeutralLosses()

    @property
    def loss_formulas(self) -> List[str]:
        """
        Get neutral loss formulas.

        Returns:
            List[str]: List of formula strings.
        """
        formulas = self._residue.getLossFormulas()
        return [f.toString() for f in formulas]

    @property
    def loss_names(self) -> List[str]:
        """Get neutral loss names."""
        return list(self._residue.getLossNames())

    @property
    def n_term_loss_formulas(self) -> List[str]:
        """
        Get N-terminal loss formulas.

        Returns:
            List[str]: List of formula strings.
        """
        formulas = self._residue.getNTermLossFormulas()
        return [f.toString() for f in formulas]

    @property
    def n_term_loss_names(self) -> List[str]:
        """Get N-terminal loss names."""
        return list(self._residue.getNTermLossNames())

    # ==================== Low Mass Ions ====================

    @property
    def low_mass_ions(self) -> List[str]:
        """
        Get low mass marker ions (e.g., immonium ions).

        Returns:
            List[str]: List of formula strings.
        """
        ions = self._residue.getLowMassIons()
        return [ion.toString() for ion in ions]

    # ==================== Physicochemical Properties ====================

    @property
    def pka(self) -> float:
        """Get pKa value."""
        return self._residue.getPka()

    @property
    def pkb(self) -> float:
        """Get pKb value."""
        return self._residue.getPkb()

    @property
    def pkc(self) -> float:
        """Get pKc value (returns -1 if not applicable)."""
        return self._residue.getPkc()

    @property
    def pi_value(self) -> float:
        """Calculate isoelectric point from pK values."""
        return self._residue.getPiValue()

    @property
    def side_chain_basicity(self) -> float:
        """Get side chain basicity (gas phase)."""
        return self._residue.getSideChainBasicity()

    @property
    def backbone_basicity_left(self) -> float:
        """Get backbone basicity in N-terminal direction."""
        return self._residue.getBackboneBasicityLeft()

    @property
    def backbone_basicity_right(self) -> float:
        """Get backbone basicity in C-terminal direction."""
        return self._residue.getBackboneBasicityRight()

    # ==================== Residue Sets ====================

    @property
    def residue_sets(self) -> Set[str]:
        """
        Get residue sets this amino acid belongs to.

        Example sets: 'Natural20', 'Natural19WithoutL', etc.

        Returns:
            Set[str]: Set of residue set names.
        """
        return self._residue.getResidueSets()

    def is_in_residue_set(self, residue_set: str) -> bool:
        """
        Check if residue is in a specific set.

        Args:
            residue_set: Name of the residue set (e.g., 'Natural20').

        Returns:
            bool: True if residue is in the set.

        Example:
            >>> ala = Py_Residue.from_one_letter_code("A")
            >>> print(ala.is_in_residue_set("Natural20"))
            True
        """
        return self._residue.isInResidueSet(residue_set)

    # ==================== Magic Methods ====================

    def __str__(self) -> str:
        """String representation using one_letter_code."""
        return self.one_letter_code

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        if self.is_modified:
            return f"Py_Residue('{self.one_letter_code}', modified='{self.modification_name}')"
        return f"Py_Residue('{self.one_letter_code}', name='{self.name}')"

    def __eq__(self, other: object) -> bool:
        """Check equality based on residue properties."""
        if isinstance(other, Py_Residue):
            return self._residue == other._residue
        elif isinstance(other, str):
            # Allow comparison with one-letter code
            return self.one_letter_code == other
        return False

    def __ne__(self, other: object) -> bool:
        """Check inequality."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Make residues hashable."""
        return hash((self.one_letter_code, self.modification_name))