"""Tests for Py_AASequence wrapper."""

from __future__ import annotations

import pytest
import pyopenms as oms

from openms_python.py_aasequence import Py_AASequence


def test_py_aasequence_from_string():
    """Test creating sequence from string."""
    seq = Py_AASequence.from_string("PEPTIDE")
    assert seq.sequence == "PEPTIDE"
    assert len(seq) == 7


def test_py_aasequence_empty():
    """Test creating empty sequence."""
    seq = Py_AASequence()
    assert len(seq) == 0
    assert seq.sequence == ""


def test_py_aasequence_properties():
    """Test basic properties."""
    seq = Py_AASequence.from_string("PEPTIDE")

    # Basic properties
    assert seq.sequence == "PEPTIDE"
    assert seq.unmodified_sequence == "PEPTIDE"
    assert len(seq) == 7

    # Weight properties
    assert seq.mono_weight > 0
    assert seq.average_weight > 0
    assert seq.mono_weight != seq.average_weight

    # Formula
    assert "C" in seq.formula
    assert "H" in seq.formula
    assert "N" in seq.formula
    assert "O" in seq.formula

    # Modification status
    assert not seq.is_modified
    assert not seq.has_n_terminal_modification
    assert not seq.has_c_terminal_modification


def test_py_aasequence_modified_sequence():
    """Test sequence with modifications."""
    # Create a modified sequence
    native_seq = oms.AASequence.fromString("PEPTIDEM(Oxidation)")
    seq = Py_AASequence(native_seq)

    assert seq.is_modified
    assert "M(Oxidation)" in seq.sequence
    assert seq.unmodified_sequence == "PEPTIDEM"


def test_py_aasequence_reverse():
    """Test reverse operation."""
    seq = Py_AASequence.from_string("PEPTIDE")
    reversed_seq = seq.reverse()

    assert reversed_seq.sequence == "EDITPEP"
    assert len(reversed_seq) == len(seq)
    # Original should be unchanged
    assert seq.sequence == "PEPTIDE"


def test_py_aasequence_reverse_with_enzyme():
    """Test reverse with enzyme constraint."""
    # Trypsin cleaves after K and R
    seq = Py_AASequence.from_string("PEPTIDERK")
    reversed_seq = seq.reverse_with_enzyme("Trypsin")

    # The sequence should be reversed in segments
    assert len(reversed_seq) == len(seq)
    # Original should be unchanged
    assert seq.sequence == "PEPTIDERK"
    # Reversed sequence should be different
    assert reversed_seq.sequence != seq.sequence


def test_py_aasequence_shuffle():
    """Test shuffle operation."""
    seq = Py_AASequence.from_string("PEPTIDERK")

    # Shuffle with a seed for reproducibility
    shuffled1 = seq.shuffle(enzyme="Trypsin", seed=42)
    shuffled2 = seq.shuffle(enzyme="Trypsin", seed=42)

    # Same seed should give same result
    assert shuffled1.sequence == shuffled2.sequence

    # Different seed should (usually) give different result
    shuffled3 = seq.shuffle(enzyme="Trypsin", seed=123)
    # Can't guarantee they're different due to randomness, but length should match
    assert len(shuffled3) == len(seq)

    # Original should be unchanged
    assert seq.sequence == "PEPTIDERK"


def test_py_aasequence_shuffle_without_seed():
    """Test shuffle without explicit seed."""
    seq = Py_AASequence.from_string("PEPTIDERK")
    shuffled = seq.shuffle(enzyme="Trypsin")

    # Should create a valid sequence of same length
    assert len(shuffled) == len(seq)


def test_py_aasequence_iteration():
    """Test iterating over residues."""
    seq = Py_AASequence.from_string("PEPTIDE")
    residues = list(seq)

    assert residues == ["P", "E", "P", "T", "I", "D", "E"]
    assert len(residues) == 7


def test_py_aasequence_indexing():
    """Test indexing into sequence."""
    seq = Py_AASequence.from_string("PEPTIDE")

    assert seq[0] == "P"
    assert seq[1] == "E"
    assert seq[6] == "E"

    # Test out of bounds
    with pytest.raises(IndexError):
        _ = seq[7]

    with pytest.raises(IndexError):
        _ = seq[-1]


def test_py_aasequence_string_representation():
    """Test string representations."""
    seq = Py_AASequence.from_string("PEPTIDE")

    assert str(seq) == "PEPTIDE"
    assert repr(seq) == "Py_AASequence('PEPTIDE')"

    # Test long sequence truncation in repr
    long_seq = Py_AASequence.from_string("PEPTIDEPEPTIDEPEPTIDEPEPTIDE")
    assert "..." in repr(long_seq)


def test_py_aasequence_equality():
    """Test equality comparison."""
    seq1 = Py_AASequence.from_string("PEPTIDE")
    seq2 = Py_AASequence.from_string("PEPTIDE")
    seq3 = Py_AASequence.from_string("DIFFERENT")

    assert seq1 == seq2
    assert seq1 != seq3
    assert seq1 != "PEPTIDE"  # Different type


def test_py_aasequence_get_mz():
    """Test m/z calculation."""
    seq = Py_AASequence.from_string("PEPTIDE")

    # Get m/z for different charge states
    mz1 = seq.get_mz(1)
    mz2 = seq.get_mz(2)
    mz3 = seq.get_mz(3)

    # Higher charge should give lower m/z
    assert mz1 > mz2 > mz3
    assert mz1 > 0
    assert mz2 > 0
    assert mz3 > 0


def test_py_aasequence_substring_operations():
    """Test substring checking."""
    seq = Py_AASequence.from_string("PEPTIDERK")

    # Test substring
    assert seq.has_substring("TIDE")
    assert seq.has_substring("PEPT")
    assert not seq.has_substring("XXX")

    # Test prefix
    assert seq.has_prefix("PEP")
    assert seq.has_prefix("PEPTIDE")
    assert not seq.has_prefix("TIDE")

    # Test suffix
    assert seq.has_suffix("RK")
    assert seq.has_suffix("DERK")
    assert not seq.has_suffix("PEP")


def test_py_aasequence_native_access():
    """Test access to native pyOpenMS object."""
    seq = Py_AASequence.from_string("PEPTIDE")
    native = seq.native

    assert isinstance(native, oms.AASequence)
    assert native.toString() == "PEPTIDE"


def test_py_aasequence_reverse_removes_modifications():
    """Test that reverse operation removes modifications (expected behavior)."""
    # Create a sequence with modification
    native_seq = oms.AASequence.fromString("PEPTIDEM(Oxidation)")
    seq = Py_AASequence(native_seq)

    assert seq.is_modified

    reversed_seq = seq.reverse()

    # DecoyGenerator removes modifications (this is expected behavior in OpenMS)
    assert not reversed_seq.is_modified
    # Original should be unchanged
    assert seq.is_modified
    # Reversed should contain the same amino acids (unmodified)
    assert "M" in reversed_seq.sequence
    assert len(reversed_seq) == len(seq)


def test_py_aasequence_different_enzymes():
    """Test different enzyme options."""
    seq = Py_AASequence.from_string("PEPTIDERK")

    # Test with Trypsin
    trypsin_rev = seq.reverse_with_enzyme("Trypsin")
    assert len(trypsin_rev) == len(seq)

    # Test with different enzyme (no cleavage)
    no_cleavage_rev = seq.reverse_with_enzyme("no cleavage")
    assert len(no_cleavage_rev) == len(seq)


def test_py_aasequence_shuffle_max_attempts():
    """Test shuffle with different max_attempts."""
    seq = Py_AASequence.from_string("PEPTIDERK")

    # Different max_attempts should still work
    shuffled1 = seq.shuffle(enzyme="Trypsin", max_attempts=10, seed=42)
    shuffled2 = seq.shuffle(enzyme="Trypsin", max_attempts=1000, seed=42)

    # Both should produce valid sequences
    assert len(shuffled1) == len(seq)
    assert len(shuffled2) == len(seq)


def test_py_aasequence_with_native_aasequence():
    """Test wrapping an existing pyOpenMS AASequence."""
    native = oms.AASequence.fromString("PEPTIDE")
    seq = Py_AASequence(native)

    assert seq.sequence == "PEPTIDE"
    assert seq.native is native
