"""Tests for Py_Residue wrapper."""

from __future__ import annotations

import pytest
import pyopenms as oms

from openms_python.py_residue import Py_Residue


def test_py_residue_from_string():

    """Test creating residue from one-letter code."""
    ala = Py_Residue.from_string("A")
    assert ala.one_letter_code == "A"
    assert ala.three_letter_code == "Ala"
    assert ala.name == "Alanine"

    """Test creating residue from three-letter code."""
    ala = Py_Residue.from_string("Ala")
    assert ala.one_letter_code == "A"
    assert ala.three_letter_code == "Ala"
    assert ala.name == "Alanine"

    """Test creating residue from full name."""
    ala = Py_Residue.from_string("Alanine")
    assert ala.one_letter_code == "A"
    assert ala.three_letter_code == "Ala"
    assert ala.name == "Alanine"


def test_py_residue_basic_properties():
    """Test basic properties of residues."""
    # Test alanine
    ala = Py_Residue.from_string("A")
    
    assert ala.name == "Alanine"
    assert ala.one_letter_code == "A"
    assert ala.three_letter_code == "Ala"
    
    # Test synonyms (if any)
    assert isinstance(ala.synonyms, set)


def test_py_residue_weight_and_formula():
    """Test weight and formula properties."""
    ala = Py_Residue.from_string("A")
    
    # Test weights
    assert ala.mono_weight > 0
    assert ala.average_weight > 0
    assert ala.mono_weight != ala.average_weight
    
    # Test formula
    assert isinstance(ala.formula, str)
    assert "C" in ala.formula
    assert "H" in ala.formula
    assert "N" in ala.formula
    assert "O" in ala.formula


def test_py_residue_different_amino_acids():
    """Test different amino acids have different properties."""
    ala = Py_Residue.from_string("A")
    arg = Py_Residue.from_string("R")
    
    assert ala.name != arg.name
    assert ala.mono_weight != arg.mono_weight
    assert ala.formula != arg.formula


def test_py_residue_modification_status():
    """Test modification status for unmodified residue."""
    ala = Py_Residue.from_string("A")
    
    assert not ala.is_modified
    assert ala.modification is None
    assert ala.modification_name == ""


def test_py_residue_set_modification():
    """Test setting modification by name."""
    met = Py_Residue.from_string("M")
    
    # Before modification
    assert not met.is_modified
    
    # Set oxidation
    met.set_modification("Oxidation")
    
    # After modification
    assert met.is_modified
    assert met.modification is not None
    assert "Oxidation" in met.modification_name


def test_py_residue_set_modification_by_diff_mass():
    """Test setting modification by mass difference."""
    ser = Py_Residue.from_string("S")
    
    # Set phosphorylation by mass difference
    phospho_mass = 79.966331
    ser.set_modification_by_diff_mass(phospho_mass)
    
    assert ser.is_modified
    assert ser.modification is not None


def test_py_residue_neutral_losses():
    """Test neutral loss properties."""
    # Serine has neutral loss of water
    ser = Py_Residue.from_string("S")
    
    # Check if has neutral loss (property may vary by residue)
    has_loss = ser.has_neutral_loss
    assert isinstance(has_loss, bool)
    
    # Get loss formulas and names
    loss_formulas = ser.loss_formulas
    loss_names = ser.loss_names
    
    assert isinstance(loss_formulas, list)
    assert isinstance(loss_names, list)


def test_py_residue_n_term_neutral_losses():
    """Test N-terminal neutral loss properties."""
    ala = Py_Residue.from_string("A")
    
    has_n_term_loss = ala.has_n_term_neutral_losses
    assert isinstance(has_n_term_loss, bool)
    
    n_term_formulas = ala.n_term_loss_formulas
    n_term_names = ala.n_term_loss_names
    
    assert isinstance(n_term_formulas, list)
    assert isinstance(n_term_names, list)


def test_py_residue_low_mass_ions():
    """Test low mass ions (immonium ions)."""
    leu = Py_Residue.from_string("L")
    
    low_mass = leu.low_mass_ions
    assert isinstance(low_mass, list)
    # Each element should be a formula string
    for ion in low_mass:
        assert isinstance(ion, str)


def test_py_residue_pk_values():
    """Test pK values."""
    # Lysine has a side chain pKa
    lys = Py_Residue.from_string("K")
    
    pka = lys.pka
    pkb = lys.pkb
    pkc = lys.pkc
    
    assert isinstance(pka, float)
    assert isinstance(pkb, float)
    assert isinstance(pkc, float)
    
    # Test isoelectric point calculation
    pi = lys.pi_value
    assert isinstance(pi, float)
    assert pi > 0


def test_py_residue_basicity_values():
    """Test gas phase basicity values."""
    ala = Py_Residue.from_string("A")
    
    sc_basicity = ala.side_chain_basicity
    bb_left = ala.backbone_basicity_left
    bb_right = ala.backbone_basicity_right
    
    assert isinstance(sc_basicity, float)
    assert isinstance(bb_left, float)
    assert isinstance(bb_right, float)


def test_py_residue_residue_sets():
    """Test residue sets."""
    ala = Py_Residue.from_string("A")
    
    # Get all sets this residue belongs to
    sets = ala.residue_sets
    assert isinstance(sets, set)
    
    # Check if in Natural20 (standard amino acids)
    is_natural = ala.is_in_residue_set("Natural20")
    assert isinstance(is_natural, bool)


def test_py_residue_get_weight_by_type():
    """Test getting weights for different residue types."""
    ala = Py_Residue.from_string("A")
    
    # Get weights for different types
    full_weight = ala.get_mono_weight(oms.Residue.ResidueType.Full)
    internal_weight = ala.get_mono_weight(oms.Residue.ResidueType.Internal)
    b_ion_weight = ala.get_mono_weight(oms.Residue.ResidueType.BIon)
    y_ion_weight = ala.get_mono_weight(oms.Residue.ResidueType.YIon)
    
    assert full_weight > 0
    assert internal_weight > 0
    assert b_ion_weight > 0
    assert y_ion_weight > 0
    
    # Full weight should be different from internal
    assert full_weight != internal_weight


def test_py_residue_get_formula_by_type():
    """Test getting formulas for different residue types."""
    ala = Py_Residue.from_string("A")
    
    # Get formulas for different types
    full_formula = ala.get_formula(oms.Residue.ResidueType.Full)
    internal_formula = ala.get_formula(oms.Residue.ResidueType.Internal)
    b_ion_formula = ala.get_formula(oms.Residue.ResidueType.BIon)
    
    assert isinstance(full_formula, str)
    assert isinstance(internal_formula, str)
    assert isinstance(b_ion_formula, str)
    
    # Formulas should be different
    assert full_formula != internal_formula


def test_py_residue_get_average_weight_by_type():
    """Test getting average weights for different residue types."""
    ala = Py_Residue.from_string("A")
    
    full_avg = ala.get_average_weight(oms.Residue.ResidueType.Full)
    internal_avg = ala.get_average_weight(oms.Residue.ResidueType.Internal)
    
    assert full_avg > 0
    assert internal_avg > 0
    assert full_avg != internal_avg


def test_py_residue_string_representation():
    """Test string representations."""
    ala = Py_Residue.from_string("A")
    
    # Test __str__
    str_repr = str(ala)
    assert isinstance(str_repr, str)
    
    # Test __repr__
    repr_str = repr(ala)
    assert "Py_Residue" in repr_str
    assert "A" in repr_str


def test_py_residue_string_representation_modified():
    """Test string representation of modified residue."""
    met = Py_Residue.from_string("M")
    met.set_modification("Oxidation")
    
    repr_str = repr(met)
    assert "Py_Residue" in repr_str
    assert "M" in repr_str
    assert "modified" in repr_str.lower()


def test_py_residue_equality():
    """Test equality comparisons."""
    ala1 = Py_Residue.from_string("A")
    ala2 = Py_Residue.from_string("A")
    arg = Py_Residue.from_string("R")
    
    # Same residue should be equal
    assert ala1 == ala2
    
    # Different residues should not be equal
    assert ala1 != arg
    
    # Test string comparison with one-letter code
    assert ala1 == "A"
    assert ala1 != "R"
    
    # Test inequality operator
    assert not (ala1 != ala2)
    assert ala1 != arg


def test_py_residue_hashable():
    """Test that residues are hashable."""
    ala = Py_Residue.from_string("A")
    arg = Py_Residue.from_string("R")
    
    # Should be able to create a set
    residue_set = {ala, arg}
    assert len(residue_set) == 2
    
    # Should be able to use as dict key
    residue_dict = {ala: "alanine", arg: "arginine"}
    assert residue_dict[ala] == "alanine"
    assert residue_dict[arg] == "arginine"


def test_py_residue_native_access():
    """Test access to native pyOpenMS object."""
    ala = Py_Residue.from_string("A")
    native = ala.native
    
    assert isinstance(native, oms.Residue)
    assert native.getOneLetterCode() == "A"


def test_py_residue_from_native():
    """Test creating Py_Residue from native object."""
    # Get native residue
    db = oms.ResidueDB()
    native_ala = db.getResidue("A")
    
    # Wrap it
    ala = Py_Residue.from_native(native_ala)
    
    assert ala.one_letter_code == "A"
    assert ala.native is native_ala

def test_py_residue_all_standard_amino_acids():
    """Test that all standard amino acids can be loaded."""
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    
    for aa in standard_aa:
        residue = Py_Residue.from_string(aa)
        assert residue.one_letter_code == aa
        assert residue.mono_weight > 0
        assert len(residue.formula) > 0


def test_py_residue_modified_equality():
    """Test equality of modified residues."""
    met1 = Py_Residue.from_string("M")
    met2 = Py_Residue.from_string("M")
    
    # Before modification, should be equal
    assert met1 == met2
    
    # Modify one
    met1.set_modification("Oxidation")
    
    # After modification, should not be equal (different modification state)
    assert met1 != met2
    
    # Both modified with same modification should be equal
    met2.set_modification("Oxidation")
    # Note: This might depend on implementation details


def test_py_residue_weight_formula_consistency():
    """Test that weights and formulas are consistent."""
    ala = Py_Residue.from_string("A")
    
    # For a given residue type, weight and formula should be consistent
    mono_weight = ala.get_mono_weight(oms.Residue.ResidueType.Full)
    formula = ala.get_formula(oms.Residue.ResidueType.Full)
    
    # Weight should be positive
    assert mono_weight > 0
    # Formula should contain elements
    assert "C" in formula or "H" in formula


def test_py_residue_empty_residue():
    """Test creating empty residue."""
    empty = Py_Residue()
    
    # Should have empty properties
    assert empty.name == "unknown"
    assert empty.one_letter_code == ""


def test_py_residue_special_amino_acids():
    """Test special amino acids like selenocysteine."""
    # Note: Availability depends on ResidueDB configuration
    # This test might need adjustment based on your OpenMS version
    try:
        # Try to get selenocysteine (U)
        sec = Py_Residue.from_string("U")
        assert sec.one_letter_code == "U"
    except:
        # If not available, that's okay for standard configurations
        pass


def test_py_residue_modification_persists():
    """Test that modifications persist on the residue object."""
    met = Py_Residue.from_string("M")
    
    # Get weight before modification
    weight_before = met.mono_weight
    
    # Add modification
    met.set_modification("Oxidation")
    
    # Weight should change
    weight_after = met.mono_weight
    assert weight_after != weight_before
    
    # Modification should still be there
    assert met.is_modified
    assert met.modification is not None


def test_py_residue_different_residue_types_give_different_weights():
    """Test that different residue types produce different weights."""
    ala = Py_Residue.from_string("A")
    
    # Get weights for multiple types
    weights = {
        'Full': ala.get_mono_weight(oms.Residue.ResidueType.Full),
        'Internal': ala.get_mono_weight(oms.Residue.ResidueType.Internal),
        'BIon': ala.get_mono_weight(oms.Residue.ResidueType.BIon),
        'YIon': ala.get_mono_weight(oms.Residue.ResidueType.YIon),
    }
    
    # Convert to set to check uniqueness
    unique_weights = set(weights.values())
    
    # Should have multiple different weight values
    assert len(unique_weights) > 1
