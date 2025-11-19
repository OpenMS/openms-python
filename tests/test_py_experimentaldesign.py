"""Tests for :class:`Py_ExperimentalDesign`."""

from pathlib import Path
import tempfile

import pandas as pd
import pytest
import pyopenms as oms

from openms_python import Py_ExperimentalDesign, get_example


def test_py_experimentaldesign_load_from_file():
    """Test loading an experimental design from the bundled example."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    assert design.n_samples == 2
    assert design.n_ms_files == 6
    assert design.n_fractions == 3
    assert design.n_fraction_groups == 2
    assert design.n_labels == 1
    assert design.is_fractionated is True


def test_py_experimentaldesign_properties():
    """Test that properties work correctly."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    # Test all properties
    assert isinstance(design.n_samples, int)
    assert isinstance(design.n_ms_files, int)
    assert isinstance(design.n_fractions, int)
    assert isinstance(design.n_fraction_groups, int)
    assert isinstance(design.n_labels, int)
    assert isinstance(design.is_fractionated, bool)
    assert isinstance(design.same_n_ms_files_per_fraction, bool)

    # Test samples property
    samples = design.samples
    assert isinstance(samples, set)
    assert len(samples) == 2
    assert "1" in samples
    assert "2" in samples


def test_py_experimentaldesign_summary():
    """Test summary method returns expected structure."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    summary = design.summary()

    assert isinstance(summary, dict)
    assert "n_samples" in summary
    assert "n_ms_files" in summary
    assert "n_fractions" in summary
    assert "n_fraction_groups" in summary
    assert "n_labels" in summary
    assert "is_fractionated" in summary
    assert "samples" in summary

    assert summary["n_samples"] == 2
    assert summary["n_ms_files"] == 6
    assert summary["samples"] == ["1", "2"]


def test_py_experimentaldesign_print_summary(capsys):
    """Test that print_summary produces output."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    design.print_summary()

    captured = capsys.readouterr()
    assert "Experimental Design Summary" in captured.out
    assert "Samples: 2" in captured.out
    assert "MS Files: 6" in captured.out
    assert "Fractionated: True" in captured.out


def test_py_experimentaldesign_load_method():
    """Test that load method works and returns self."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign()

    result = design.load(example_path)

    assert result is design
    assert design.n_samples == 2


def test_py_experimentaldesign_native_property():
    """Test that native property returns pyopenms object."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    native = design.native

    assert isinstance(native, oms.ExperimentalDesign)
    assert native.getNumberOfSamples() == 2


def test_py_experimentaldesign_repr():
    """Test string representation."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    repr_str = repr(design)

    assert "Py_ExperimentalDesign" in repr_str
    assert "samples=2" in repr_str
    assert "ms_files=6" in repr_str
    assert "fractionated=True" in repr_str


def test_py_experimentaldesign_invalid_extension():
    """Test that loading a file with wrong extension raises error."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("test")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="ExperimentalDesign"):
            Py_ExperimentalDesign.from_file(temp_path)
    finally:
        Path(temp_path).unlink()


def test_py_experimentaldesign_store_roundtrip():
    """Test that store method works and roundtrips correctly."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
        temp_path = f.name

    try:
        # Store the design
        result = design.store(temp_path)
        assert result is design  # Check method chaining

        # Verify file was created
        assert Path(temp_path).exists()

        # Load it back and verify
        design2 = Py_ExperimentalDesign.from_file(temp_path)
        assert design2.n_samples == design.n_samples
        assert design2.n_ms_files == design.n_ms_files
        assert design2.n_fractions == design.n_fractions
        assert design2.n_fraction_groups == design.n_fraction_groups
        assert design2.n_labels == design.n_labels

        # Verify DataFrame content matches
        df1 = design.to_dataframe()
        df2 = design2.to_dataframe()
        assert df1.equals(df2)
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def test_py_experimentaldesign_store_invalid_extension():
    """Test that store rejects invalid file extensions."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="ExperimentalDesign"):
            design.store(temp_path)
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def test_py_experimentaldesign_delegation():
    """Test that methods are delegated to underlying object."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    # Test delegation of a native method
    ms_file_section = design.getMSFileSection()
    assert isinstance(ms_file_section, list)
    assert len(ms_file_section) == 6


def test_py_experimentaldesign_simple_design():
    """Test with a simple non-fractionated design."""
    # Create a simple experimental design
    # Different fraction groups for different samples to avoid conflicts
    tsv_content = """Fraction_Group\tFraction\tSpectra_Filepath\tLabel\tSample
1\t1\tfile1.mzML\t1\t1
2\t1\tfile2.mzML\t1\t2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write(tsv_content)
        temp_path = f.name

    try:
        design = Py_ExperimentalDesign.from_file(temp_path)

        assert design.n_samples == 2
        assert design.n_ms_files == 2
        assert design.n_fractions == 1
        assert design.n_labels == 1

        samples = design.samples
        assert len(samples) == 2

    finally:
        Path(temp_path).unlink()


def test_py_experimentaldesign_from_consensus_map():
    """Test creating ExperimentalDesign from ConsensusMap."""
    # Create a minimal consensus map
    consensus_map = oms.ConsensusMap()

    # Add some file descriptions
    file_desc1 = oms.ProteinIdentification()
    file_desc1.setIdentifier("file1")
    file_desc2 = oms.ProteinIdentification()
    file_desc2.setIdentifier("file2")

    consensus_map.setProteinIdentifications([file_desc1, file_desc2])

    # Create design from consensus map
    design = Py_ExperimentalDesign.from_consensus_map(consensus_map)

    assert isinstance(design, Py_ExperimentalDesign)
    assert isinstance(design.native, oms.ExperimentalDesign)


def test_py_experimentaldesign_from_feature_map():
    """Test creating ExperimentalDesign from FeatureMap."""
    # Create a minimal feature map
    feature_map = oms.FeatureMap()

    # Add protein identification for the file descriptor
    prot_id = oms.ProteinIdentification()
    prot_id.setIdentifier("sample1")
    feature_map.setProteinIdentifications([prot_id])

    # Create design from feature map
    design = Py_ExperimentalDesign.from_feature_map(feature_map)

    assert isinstance(design, Py_ExperimentalDesign)
    assert isinstance(design.native, oms.ExperimentalDesign)


def test_py_experimentaldesign_from_identifications():
    """Test creating ExperimentalDesign from identification data."""
    # Create minimal identification data
    prot_id = oms.ProteinIdentification()
    prot_id.setIdentifier("search1")

    protein_ids = [prot_id]

    # Create design from identifications
    design = Py_ExperimentalDesign.from_identifications(protein_ids)

    assert isinstance(design, Py_ExperimentalDesign)
    assert isinstance(design.native, oms.ExperimentalDesign)


def test_py_experimentaldesign_from_dataframe():
    """Test creating ExperimentalDesign from a pandas DataFrame."""
    import pandas as pd

    # Create a simple DataFrame
    df = pd.DataFrame(
        {
            "Fraction_Group": [1, 1, 2, 2],
            "Fraction": [1, 2, 1, 2],
            "Spectra_Filepath": ["f1.mzML", "f2.mzML", "f3.mzML", "f4.mzML"],
            "Label": [1, 1, 1, 1],
            "Sample": [1, 1, 2, 2],
        }
    )

    design = Py_ExperimentalDesign.from_dataframe(df)

    assert isinstance(design, Py_ExperimentalDesign)
    assert design.n_ms_files == 4
    assert design.n_samples == 2
    assert design.n_fraction_groups == 2


def test_py_experimentaldesign_from_df_alias():
    """Test that from_df is an alias for from_dataframe."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "Fraction_Group": [1],
            "Fraction": [1],
            "Spectra_Filepath": ["test.mzML"],
            "Label": [1],
            "Sample": [1],
        }
    )

    design1 = Py_ExperimentalDesign.from_dataframe(df)
    design2 = Py_ExperimentalDesign.from_df(df)

    assert design1.n_ms_files == design2.n_ms_files


def test_py_experimentaldesign_from_dataframe_missing_columns():
    """Test that from_dataframe raises error for missing columns."""
    import pandas as pd

    # Missing Sample column
    df = pd.DataFrame(
        {
            "Fraction_Group": [1],
            "Fraction": [1],
            "Spectra_Filepath": ["test.mzML"],
            "Label": [1],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        Py_ExperimentalDesign.from_dataframe(df)


def test_py_experimentaldesign_to_dataframe():
    """Test converting ExperimentalDesign to DataFrame."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    df = design.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert set(df.columns) == {
        "Fraction_Group",
        "Fraction",
        "Spectra_Filepath",
        "Label",
        "Sample",
    }
    assert df["Sample"].nunique() == 2


def test_py_experimentaldesign_get_df_alias():
    """Test that get_df is an alias for to_dataframe."""
    example_path = get_example("experimental_design.tsv")
    design = Py_ExperimentalDesign.from_file(example_path)

    df1 = design.to_dataframe()
    df2 = design.get_df()

    assert df1.equals(df2)


def test_py_experimentaldesign_dataframe_roundtrip():
    """Test that DataFrame roundtrip preserves data."""
    import pandas as pd

    # Create a DataFrame with specific data
    original_df = pd.DataFrame(
        {
            "Fraction_Group": [1, 1, 2],
            "Fraction": [1, 2, 1],
            "Spectra_Filepath": ["a.mzML", "b.mzML", "c.mzML"],
            "Label": [1, 1, 2],
            "Sample": [1, 1, 2],
        }
    )

    # Create design from DataFrame
    design = Py_ExperimentalDesign.from_dataframe(original_df)

    # Convert back to DataFrame
    result_df = design.to_dataframe()

    # Check that data is preserved
    assert len(result_df) == len(original_df)
    assert result_df["Fraction_Group"].tolist() == original_df["Fraction_Group"].tolist()
    assert result_df["Fraction"].tolist() == original_df["Fraction"].tolist()
    assert result_df["Spectra_Filepath"].tolist() == original_df["Spectra_Filepath"].tolist()
    assert result_df["Label"].tolist() == original_df["Label"].tolist()
    assert result_df["Sample"].tolist() == original_df["Sample"].tolist()
