#!/usr/bin/env python3
"""Warmup script to pre-compile numba functions with minimal memory."""

print("🔥 Warming up numba/librosa with minimal memory...")

import numpy as np
import librosa
import os

# Disable numba caching to save memory during build
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

# Create small dummy audio to trigger compilation (use less memory)
dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

print("📊 Testing pitch detection...")
pitches, magnitudes = librosa.piptrack(y=dummy_audio, sr=16000, n_fft=1024, hop_length=512)

print("📊 Testing tempo...")
tempo, _ = librosa.beat.beat_track(y=dummy_audio, sr=16000, hop_length=512)

print("📊 Testing spectral features...")
spectral_centroid = librosa.feature.spectral_centroid(y=dummy_audio, sr=16000, n_fft=1024, hop_length=512)

# Clean up to free memory
del dummy_audio, pitches, magnitudes, spectral_centroid

print("✅ Warmup complete! Numba functions compiled with minimal memory footprint.")

