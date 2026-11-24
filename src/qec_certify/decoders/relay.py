"""Relay-BP decoder on arbitrary detector error models."""

from __future__ import annotations

import numpy as np
import stim
from relay_bp import ObservableDecoderRunner, RelayDecoderF32, RelayDecoderI32, RelayDecoderI64

from qec_certify.decoders.dem import DEMDecoder


class RelayBPDecoder(DEMDecoder):
    """Public Rust Relay-BP implementation with DEM probabilities as error priors.

    Unsuccessful decodes fall back to the deterministic no-flip frame and are exposed through
    ``last_decode_success``; an unsuccessful candidate is never returned as a valid correction.
    Serial prediction constructs a fresh native decoder per shot to prevent state carryover;
    parallel prediction uses the public runner's independent clones.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        gamma0: float = 0.1,
        pre_iter: int = 80,
        num_sets: int = 60,
        set_max_iter: int = 60,
        gamma_dist_interval: tuple[float, float] = (-0.24, 0.66),
        stop_nconv: int = 1,
        seed: int = 0,
        implementation: str = "f32",
        data_scale_value: float | None = None,
        max_data_value: float | None = None,
        parallel: bool = False,
    ) -> None:
        super().__init__(dem)
        decoder_class = {
            "f32": RelayDecoderF32,
            "i32": RelayDecoderI32,
            "i64": RelayDecoderI64,
        }.get(implementation)
        if decoder_class is None:
            raise ValueError(
                f"unknown Relay-BP implementation {implementation!r}; "
                "choose from ['f32', 'i32', 'i64']"
            )
        self.implementation = implementation
        self.parallel = parallel
        options = {
            "gamma0": gamma0,
            "pre_iter": pre_iter,
            "num_sets": num_sets,
            "set_max_iter": set_max_iter,
            "gamma_dist_interval": gamma_dist_interval,
            "stop_nconv": stop_nconv,
            "seed": seed,
        }
        if data_scale_value is not None:
            options["data_scale_value"] = data_scale_value
        if max_data_value is not None:
            options["max_data_value"] = max_data_value
        self._decoder_factory = lambda: decoder_class(
            self.check_matrix,
            error_priors=self.channel_probabilities,
            **options,
        )
        self.decoder = self._decoder_factory() if self.num_mechanisms else None
        self.runner = (
            ObservableDecoderRunner(
                self.decoder, self.observable_matrix, include_decode_result=True
            )
            if self.num_mechanisms
            else None
        )

    def _decode(self, syndrome: np.ndarray) -> np.ndarray:
        assert self.decoder is not None
        return self.decoder.decode(syndrome)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = self._check_detection_events(detection_events)
        self.last_decode_success = np.ones(len(dets), dtype=bool)
        self.last_decode_iterations = np.zeros(len(dets), dtype=np.int64)
        if not self.num_mechanisms:
            return np.tile(self._constant_observables, (len(dets), 1))
        syndromes = dets.astype(np.uint8) ^ self._constant_syndrome
        if self.parallel:
            results = self.runner.decode_observables_detailed_batch(
                syndromes, parallel=True, progress_bar=False
            )
            logical = np.asarray([result.observables for result in results], dtype=np.uint8)
            self.last_decode_success = np.asarray(
                [result.converged for result in results], dtype=bool
            )
        else:
            results = [self._decoder_factory().decode_detailed(syndrome) for syndrome in syndromes]
            mechanisms = np.asarray([result.decoding for result in results], dtype=np.uint8)
            self.last_decode_success = np.asarray(
                [result.success for result in results], dtype=bool
            )
            logical = np.asarray(mechanisms @ self.observable_matrix.T).astype(np.uint8) % 2
        self.last_decode_iterations = np.asarray(
            [result.iterations for result in results], dtype=np.int64
        )
        logical = logical.astype(bool) ^ self._constant_observables
        logical[~self.last_decode_success] = self._constant_observables
        return logical
