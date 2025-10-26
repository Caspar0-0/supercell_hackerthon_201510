#!/usr/bin/env python3
"""Warmup script to pre-compile numba functions."""

print("🔥 Warming up numba/librosa...")

import numpy as np
import librosa

# Create dummy audio to trigger numba compilation
dummy_audio = np.random.randn(22050 * 2)  # 2 seconds of audio

print("📊 Testing pitch detection...")
pitches, magnitudes = librosa.piptrack(y=dummy_audio, sr=22050)

print("📊 Testing tempo...")
tempo, _ = librosa.beat.beat_track(y=dummy_audio, sr=22050)

print("📊 Testing spectral features...")
spectral_centroid = librosa.feature.spectral_centroid(y=dummy_audio, sr=22050)

print("✅ Warmup complete! Numba functions are now compiled.")

