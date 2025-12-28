import logging

import numpy as np

try:
    from openwakeword import Model
except ImportError:
    Model = None
    logging.warning(
        "openwakeword not available. Install with: pip install openwakeword"
    )

from client.config.config import WakeWordConfig

logger = logging.getLogger(__name__)

# OpenWakeWord works optimally with 1280 samples (80ms at 16kHz)
# We buffer smaller chunks until we have enough samples
OPTIMAL_CHUNK_SIZE = 1280


class WakeWordDetector:

    def __init__(self, config: WakeWordConfig, sample_rate: int = 16000):
        if Model is None:
            raise ImportError(
                "openwakeword is not installed. Install with: pip install openwakeword"
            )

        self.config = config
        self.sample_rate = sample_rate

        if sample_rate != 16000:
            raise ValueError(
                f"OpenWakeWord requires 16kHz sample rate, got {sample_rate}"
            )

        # Initialize OpenWakeWord model
        try:
            self.model = Model(wakeword_models=[config.model_name])
            logger.info(
                "WakeWordDetector initialized: model=%s threshold=%.2f",
                config.model_name,
                config.threshold,
            )
        except Exception as e:
            logger.error("Failed to load wake word model: %s", e)
            raise

        # Buffer to accumulate audio chunks until we have optimal size
        self._audio_buffer = np.array([], dtype=np.float32)

        # State tracking
        self._last_detection_time = 0.0
        self._activation_delay_seconds = config.activation_delay_ms / 1000.0

        # Debug counters
        self._frames_processed = 0
        self._predictions_made = 0
        self._last_log_time = 0.0

    def process_frame(self, audio_float32: np.ndarray) -> bool:
        """Process an audio frame for wake word detection.

        Args:
            audio_float32: Audio chunk as float32 numpy array (-1.0 to 1.0)

        Returns:
            True if wake word detected, False otherwise
        """
        import time

        try:
            self._frames_processed += 1

            # Log first frame and periodically to show we're receiving audio
            if self._frames_processed == 1:
                logger.info(
                    "First audio frame received: shape=%s dtype=%s samples=%d min=%.4f max=%.4f",
                    audio_float32.shape,
                    audio_float32.dtype,
                    len(audio_float32),
                    np.min(audio_float32),
                    np.max(audio_float32),
                )
            elif (
                self._frames_processed % 50 == 0
            ):  # Log every 50 frames (~1 second at 20ms chunks)
                current_time = time.time()
                if current_time - self._last_log_time > 2.0:  # Log every 2 seconds max
                    logger.info(
                        "Processing audio: frames=%d buffer_size=%d samples audio_range=[%.4f, %.4f]",
                        self._frames_processed,
                        len(self._audio_buffer),
                        np.min(audio_float32) if len(audio_float32) > 0 else 0.0,
                        np.max(audio_float32) if len(audio_float32) > 0 else 0.0,
                    )
                    self._last_log_time = current_time

            # Ensure it's a 1D array
            original_shape = audio_float32.shape
            if audio_float32.ndim > 1:
                audio_float32 = audio_float32.flatten()
                logger.debug(
                    "Flattened audio: original_shape=%s new_shape=%s",
                    original_shape,
                    audio_float32.shape,
                )

            # Add to buffer
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_float32])

            # Process when we have enough samples (optimal is 1280, but can process smaller)
            # We process when buffer >= 320 samples (20ms) to avoid too much delay
            min_chunk_size = 320  # 20ms minimum
            if len(self._audio_buffer) < min_chunk_size:
                if self._frames_processed <= 5:  # Log first few frames
                    logger.debug(
                        "Buffering audio: buffer=%d samples (need %d), chunk=%d samples",
                        len(self._audio_buffer),
                        min_chunk_size,
                        len(audio_float32),
                    )
                return False

            # Process accumulated audio
            # OpenWakeWord documentation suggests it works with int16 PCM audio
            # Convert float32 (-1.0 to 1.0) to int16 (-32768 to 32767)
            audio_clipped = np.clip(self._audio_buffer, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)

            # Log processing details periodically (less frequently)
            if self._predictions_made == 0 or self._predictions_made % 100 == 0:
                logger.info(
                    "Processing wake word detection: buffer_samples=%d float_range=[%.4f, %.4f] int16_range=[%d, %d]",
                    len(audio_int16),
                    np.min(audio_clipped),
                    np.max(audio_clipped),
                    np.min(audio_int16),
                    np.max(audio_int16),
                )

            # Process with OpenWakeWord
            # predict returns a dict with model names as keys and confidence scores as values
            # Use int16 format as per OpenWakeWord examples
            predictions = self.model.predict(audio_int16)
            self._predictions_made += 1

            # Clear buffer after processing (but keep some overlap for continuity)
            # Keep last 320 samples for overlap
            overlap_size = 320
            if len(self._audio_buffer) > overlap_size:
                self._audio_buffer = self._audio_buffer[-overlap_size:]
            else:
                self._audio_buffer = np.array([], dtype=np.float32)

            # Log predictions only when there's significant activity or periodically
            if predictions:
                # Convert numpy float32 values to regular floats for max calculation
                confidence_values = [float(v) for v in predictions.values()]
                max_confidence = max(confidence_values) if confidence_values else 0.0

                # Log only when confidence > 0.01 (1%) or every 50 predictions
                if max_confidence > 0.01 or self._predictions_made % 50 == 0:
                    logger.info(
                        "Wake word predictions (frame %d): %s (max=%.6f, threshold=%.2f)",
                        self._predictions_made,
                        {k: float(v) for k, v in predictions.items()},
                        max_confidence,
                        self.config.threshold,
                    )
            else:
                # Log when predictions dict is empty
                if self._predictions_made % 100 == 0:  # Log less frequently
                    logger.debug(
                        "No predictions returned (frame %d), buffer_size=%d",
                        self._predictions_made,
                        len(audio_clipped),
                    )

            # Check if any wake word model detected above threshold
            if predictions:
                for model_name, confidence in predictions.items():
                    # Convert numpy float32 to regular float for comparison
                    conf_value = float(confidence)
                    if conf_value >= self.config.threshold:
                        logger.info(
                            "*** WAKE WORD DETECTED ***: %s (confidence=%.4f >= threshold %.2f)",
                            model_name,
                            conf_value,
                            self.config.threshold,
                        )
                        # Clear buffer on detection
                        self._audio_buffer = np.array([], dtype=np.float32)
                        return True

            return False

        except Exception as e:
            logger.exception(
                "Error processing wake word frame (frame %d): %s",
                self._frames_processed,
                e,
            )
            return False

    def reset(self) -> None:
        """Reset the wake word detector state."""
        logger.debug("Wake word detector reset")
        self._last_detection_time = 0.0
        self._audio_buffer = np.array([], dtype=np.float32)
        # OpenWakeWord models maintain their own internal state
        if hasattr(self.model, "reset"):
            self.model.reset()

    def close(self) -> None:
        """Clean up resources."""
        # OpenWakeWord models don't require explicit cleanup, but we can
        # reset state if needed
        self.reset()
