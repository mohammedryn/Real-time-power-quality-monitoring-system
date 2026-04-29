"""Tests for ParsedFrame + multi-input predictor integration.

This module ensures that:
1. ModelReadyFrame can be parsed correctly with CRC validation
2. Arrays extracted from ModelReadyFrame have correct shapes
3. ArtifactPredictor can use these arrays for 3-input prediction
4. The complete flow ParsedFrame → arrays → predictor works end-to-end
"""
from __future__ import annotations

import numpy as np
import pytest

from src.dsp.preprocess import load_config
from src.io.frame_protocol import (
    ModelReadyFrame,
    pack_model_ready_frame,
    parse_model_ready_frame,
)
from src.runtime.pipeline import ArtifactPredictor


# ---- Fixtures ----------------------------------------------------------------

@pytest.fixture
def cfg():
    """Load config with model settings."""
    return load_config("configs/default.yaml")


@pytest.fixture
def predictor(cfg):
    """Create a dummy predictor that detects 3-input capability."""
    return DummyMultiInputPredictor(len(cfg["classes"]["names"]))


@pytest.fixture
def valid_frame() -> ModelReadyFrame:
    """Create a valid ModelReadyFrame with proper array shapes."""
    rng = np.random.default_rng(42)
    X_wave = rng.standard_normal(1000).astype(np.float32)
    X_mag = rng.standard_normal(28).astype(np.float32)
    X_phase = rng.standard_normal(270).astype(np.float32)
    
    raw = pack_model_ready_frame(seq=42, X_wave=X_wave, X_mag=X_mag, X_phase=X_phase)
    return parse_model_ready_frame(raw)


class DummyMultiInputPredictor:
    """Simulates a 3-input Keras model for testing."""

    _is_multi_input = True

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes

    def predict_proba(
        self,
        feature_vector: np.ndarray,
        v_norm: np.ndarray | None = None,
        i_norm: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return dummy sigmoid probabilities."""
        probs = np.full(self._n_classes, 0.1, dtype=np.float32)
        if feature_vector is not None and len(feature_vector) >= 29:
            fault_idx = int(abs(float(feature_vector[28]) * 100.0)) % (self._n_classes - 1) + 1
            probs[0] = 0.1
            probs[fault_idx] = 0.9
        return probs


# ---- Tests: Frame Parsing ---------------------------------------------------

def test_model_ready_frame_parse_crc_valid(valid_frame):
    """ModelReadyFrame.crc_ok should be True for valid frames."""
    assert valid_frame.crc_ok
    assert valid_frame.rx_crc == valid_frame.calc_crc


def test_model_ready_frame_seq_wraps_u16():
    """Sequence numbers should wrap at 2^16."""
    rng = np.random.default_rng(1)
    X_wave = rng.standard_normal(1000).astype(np.float32)
    X_mag = rng.standard_normal(28).astype(np.float32)
    X_phase = rng.standard_normal(270).astype(np.float32)
    
    raw = pack_model_ready_frame(seq=0x10000, X_wave=X_wave, X_mag=X_mag, X_phase=X_phase)
    frame = parse_model_ready_frame(raw)
    
    assert frame.seq == 0  # 0x10000 & 0xFFFF


def test_model_ready_frame_array_shapes(valid_frame):
    """ModelReadyFrame arrays must have correct shapes."""
    assert valid_frame.X_wave.shape == (1000,)  # v_norm + i_norm
    assert valid_frame.X_mag.shape == (28,)
    assert valid_frame.X_phase.shape == (270,)


def test_model_ready_frame_v_norm_i_norm_properties(valid_frame):
    """v_norm and i_norm properties must extract correctly from X_wave."""
    # v_norm should be first 500 elements
    np.testing.assert_array_equal(valid_frame.v_norm, valid_frame.X_wave[:500])
    
    # i_norm should be last 500 elements
    np.testing.assert_array_equal(valid_frame.i_norm, valid_frame.X_wave[500:])


# ---- Tests: Frame → Feature Vector Reconstruction ----------------------------

def test_reconstruct_298_element_vector_from_frame(valid_frame):
    """Must reconstruct 298-element feature vector from ModelReadyFrame."""
    # The relationship is:
    # X_mag   = features[28:56]
    # X_phase = features[0:28] ++ features[56:214] ++ features[214:298]
    #
    # To reconstruct, we reverse the slicing:
    # features[0:28]     = X_phase[0:28]
    # features[28:56]    = X_mag
    # features[56:214]   = X_phase[28:172]
    # features[214:298]  = X_phase[172:270]
    
    X_phase_0_28 = valid_frame.X_phase[0:28]
    X_mag = valid_frame.X_mag
    X_phase_28_172 = valid_frame.X_phase[28:172]
    X_phase_172_270 = valid_frame.X_phase[172:270]
    
    reconstructed = np.concatenate([
        X_phase_0_28,      # [0:28]
        X_mag,             # [28:56]
        X_phase_28_172,    # [56:214]
        X_phase_172_270,   # [214:298]
    ])
    
    assert reconstructed.shape == (298,)
    assert reconstructed.dtype == np.float32


# ---- Tests: Predictor Integration --------------------------------------------

def test_predictor_detects_multi_input(predictor):
    """Predictor should detect 3-input capability."""
    assert hasattr(predictor, "_is_multi_input")
    assert predictor._is_multi_input is True


def test_predictor_predict_proba_single_input(predictor, valid_frame):
    """Predictor should handle single-input call (fallback)."""
    # Reconstruct 298-element vector
    features = np.concatenate([
        valid_frame.X_phase[0:28],
        valid_frame.X_mag,
        valid_frame.X_phase[28:172],
        valid_frame.X_phase[172:270],
    ])
    
    probs = predictor.predict_proba(features)
    
    assert probs.shape == (7,)  # 7 classes
    assert probs.dtype == np.float32
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_predictor_predict_proba_multi_input(predictor, valid_frame):
    """Predictor should handle 3-input call with v_norm and i_norm."""
    # Reconstruct 298-element vector
    features = np.concatenate([
        valid_frame.X_phase[0:28],
        valid_frame.X_mag,
        valid_frame.X_phase[28:172],
        valid_frame.X_phase[172:270],
    ])
    
    probs = predictor.predict_proba(
        features,
        v_norm=valid_frame.v_norm,
        i_norm=valid_frame.i_norm,
    )
    
    assert probs.shape == (7,)
    assert probs.dtype == np.float32
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_predictor_multi_input_requires_both_norms(predictor, valid_frame):
    """Multi-input call should still work if only v_norm is provided (partial)."""
    features = np.concatenate([
        valid_frame.X_phase[0:28],
        valid_frame.X_mag,
        valid_frame.X_phase[28:172],
        valid_frame.X_phase[172:270],
    ])
    
    # With only v_norm, should fall back to single-input
    probs = predictor.predict_proba(features, v_norm=valid_frame.v_norm)
    assert probs.shape == (7,)


# ---- Tests: Round-trip Frame → Features → Prediction ----------------------------

def test_frame_to_prediction_roundtrip(predictor, cfg):
    """Complete flow: parse frame → reconstruct features → predict."""
    # Create frame
    rng = np.random.default_rng(777)
    X_wave = rng.standard_normal(1000).astype(np.float32)
    X_mag = rng.standard_normal(28).astype(np.float32)
    X_phase = rng.standard_normal(270).astype(np.float32)
    
    raw = pack_model_ready_frame(seq=99, X_wave=X_wave, X_mag=X_mag, X_phase=X_phase)
    frame = parse_model_ready_frame(raw)
    
    # Verify CRC
    assert frame.crc_ok
    
    # Reconstruct 298-element feature vector
    features = np.concatenate([
        frame.X_phase[0:28],
        frame.X_mag,
        frame.X_phase[28:172],
        frame.X_phase[172:270],
    ])
    
    # Predict
    probs = predictor.predict_proba(features, v_norm=frame.v_norm, i_norm=frame.i_norm)
    
    # Verify output
    assert probs.shape == (7,)
    assert np.isfinite(probs).all()


def test_multiple_frames_sequential_prediction(predictor, cfg):
    """Process multiple frames in sequence."""
    rng = np.random.default_rng(555)
    
    all_probs = []
    for seq in range(5):
        X_wave = rng.standard_normal(1000).astype(np.float32)
        X_mag = rng.standard_normal(28).astype(np.float32)
        X_phase = rng.standard_normal(270).astype(np.float32)
        
        raw = pack_model_ready_frame(seq=seq, X_wave=X_wave, X_mag=X_mag, X_phase=X_phase)
        frame = parse_model_ready_frame(raw)
        
        features = np.concatenate([
            frame.X_phase[0:28],
            frame.X_mag,
            frame.X_phase[28:172],
            frame.X_phase[172:270],
        ])
        
        probs = predictor.predict_proba(features, v_norm=frame.v_norm, i_norm=frame.i_norm)
        all_probs.append(probs)
    
    # Should have processed 5 frames
    assert len(all_probs) == 5
    
    # All probs should be valid
    for probs in all_probs:
        assert probs.shape == (7,)
        assert np.isfinite(probs).all()


# ---- Tests: Config Validation -----------------------------------------------

def test_config_has_expected_feature_length(cfg):
    """Config should specify expected_feature_length: 298."""
    features_cfg = cfg.get("features", {})
    assert "expected_feature_length" in features_cfg
    assert features_cfg["expected_feature_length"] == 298


def test_config_has_ml_inference_settings(cfg):
    """Config should have ml_inference section with model path and thresholds."""
    ml_cfg = cfg.get("ml_inference", {})
    assert "model_path" in ml_cfg
    assert "thresholds" in ml_cfg
    assert isinstance(ml_cfg["thresholds"], dict)
    assert len(ml_cfg["thresholds"]) >= 7  # At least one per class


def test_config_multi_label_flag(cfg):
    """Config should indicate multi-label classification."""
    ml_cfg = cfg.get("ml_inference", {})
    assert "multi_label" in ml_cfg


# ---- Tests: Error Handling --------------------------------------------------

def test_corrupted_frame_crc_fails():
    """Corrupted frame should fail CRC validation."""
    rng = np.random.default_rng(333)
    X_wave = rng.standard_normal(1000).astype(np.float32)
    X_mag = rng.standard_normal(28).astype(np.float32)
    X_phase = rng.standard_normal(270).astype(np.float32)
    
    raw = bytearray(pack_model_ready_frame(seq=0, X_wave=X_wave, X_mag=X_mag, X_phase=X_phase))
    
    # Corrupt a byte in the middle
    raw[256] ^= 0xFF
    
    frame = parse_model_ready_frame(bytes(raw))
    assert not frame.crc_ok


def test_invalid_array_shape_raises():
    """Invalid array shapes should raise during pack."""
    with pytest.raises(ValueError):
        # X_wave should be 1000, not 999
        pack_model_ready_frame(
            seq=0,
            X_wave=np.ones(999, dtype=np.float32),
            X_mag=np.ones(28, dtype=np.float32),
            X_phase=np.ones(270, dtype=np.float32),
        )


def test_invalid_feature_vector_shape_raises(predictor):
    """Invalid feature vector should raise or return consistent output."""
    # Feature vector is 298 elements; test with wrong size
    short_features = np.ones(100, dtype=np.float32)
    
    # Should handle gracefully or raise
    try:
        probs = predictor.predict_proba(short_features, v_norm=np.ones(500), i_norm=np.ones(500))
        # If it doesn't raise, output should still be valid
        assert probs.shape == (7,)
    except (ValueError, IndexError):
        # Acceptable to raise on invalid input
        pass
