import pyopenms as oms
import pytest

from openms_python import Py_FeatureMap, Py_ConsensusMap


def _simple_feature_map(rt: float) -> oms.FeatureMap:
    fmap = oms.FeatureMap()
    feature = oms.Feature()
    feature.setRT(rt)
    feature.setMZ(500.0)
    feature.setIntensity(100.0)
    fmap.push_back(feature)
    return fmap


def test_align_and_link_identity_creates_consensus():
    fmap_a = Py_FeatureMap(_simple_feature_map(10.0))
    fmap_b = Py_FeatureMap(_simple_feature_map(10.0))

    consensus = Py_ConsensusMap.align_and_link(
        [fmap_a, fmap_b],
        alignment_method="identity",
    )

    assert isinstance(consensus, Py_ConsensusMap)
    assert len(consensus) == 1
    headers = consensus.native.getColumnHeaders()
    assert len(headers) == 2


def test_align_and_link_invalid_method_raises():
    fmap = Py_FeatureMap(_simple_feature_map(10.0))
    with pytest.raises(ValueError):
        Py_ConsensusMap.align_and_link([fmap], alignment_method="unknown")


def test_align_and_link_with_kd_grouping():
    """Test that KD-tree grouping method is supported."""
    fmap_a = Py_FeatureMap(_simple_feature_map(10.0))
    fmap_b = Py_FeatureMap(_simple_feature_map(10.0))

    consensus = Py_ConsensusMap.align_and_link(
        [fmap_a, fmap_b],
        alignment_method="identity",
        grouping_method="kd",
    )

    assert isinstance(consensus, Py_ConsensusMap)
    assert len(consensus) == 1


def test_align_and_link_invalid_grouping_raises():
    """Test that invalid grouping method raises error."""
    fmap = Py_FeatureMap(_simple_feature_map(10.0))
    with pytest.raises(ValueError, match="Unsupported grouping_method"):
        Py_ConsensusMap.align_and_link(
            [fmap], 
            alignment_method="identity",
            grouping_method="invalid"
        )


def test_align_and_link_with_grouping_params():
    """Test that grouping parameters can be passed."""
    fmap_a = Py_FeatureMap(_simple_feature_map(10.0))
    fmap_b = Py_FeatureMap(_simple_feature_map(10.0))

    consensus = Py_ConsensusMap.align_and_link(
        [fmap_a, fmap_b],
        alignment_method="identity",
        grouping_method="qt",
        grouping_params={"distance_RT:max_difference": 100.0},
    )

    assert isinstance(consensus, Py_ConsensusMap)
    assert len(consensus) == 1
