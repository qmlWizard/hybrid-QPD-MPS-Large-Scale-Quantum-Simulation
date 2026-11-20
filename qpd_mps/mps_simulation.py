"""
MPSSimulationAnalyzer (QPD-aware)
=================================
Analyses a circuit produced by `MitaraiFujiDecomposition.decompose()` with Qiskit
Aer's matrix_product_state method and compares it to the ORIGINAL (uncut) circuit.

Why this differs from the previous version
------------------------------------------
A decomposed circuit samples one random branch per shot, so:
  * MPS statistics (bond dimension, memory) are TRAJECTORY dependent -> we run many
    single-shot trajectories and aggregate them.
  * The state is saved just BEFORE the final output measurements. Otherwise the
    measurement collapses the MPS to a product state (bond dimension 1).
  * The observable is NOT a plain expectation value of one trajectory. It is
    estimated by `decomposer.reconstruct(...)` (kappa * sign * beta weighting).
  * Error is measured against the original uncut circuit, not against the same
    decomposed circuit without truncation.

Assumptions
-----------
  * Observable is a Z/I Pauli string (diagonal), as required by `reconstruct`.
  * Original circuit maps qubit i -> clbit i for the output measurements
    (true for `measure_all()`), so labels line up with Statevector ordering.
  * The decomposer's extra ancilla `vg_rng` is the LAST qubit; its bond is excluded
    from bond-dimension statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

try:
    import psutil
except ImportError:
    psutil = None


# --------------------------------------------------------------------------- #
@dataclass
class MPSStats:
    """Aggregate MPS statistics over one or more trajectories."""
    n_trajectories: int
    max_bond_dimension: int            # max over trajectories of the max bond
    mean_max_bond_dimension: float
    mean_avg_bond_dimension: float
    mean_bond_profile: List[float]     # mean dimension of each bond
    mean_memory_bytes: float
    peak_memory_bytes: int
    std_max_bond_dimension: float = 0.0


@dataclass
class MPSReport:
    n_qubits: int
    original: MPSStats
    decomposed: MPSStats
    bond_reduction_max: float          # original max bond / decomposed max bond (max over trajectories)
    memory_reduction: float            # original memory / mean decomposed memory
    rss_delta_bytes: Optional[int]
    # QPD
    kappa: float
    qpd_sampling_overhead: float       # kappa^2
    optimal_shots: int
    shots_used: int
    shots_capped: bool
    num_cut_rotations: int
    # observable
    reconstructed_value: float
    reconstruction_std_error: float
    reference_value: Optional[float]
    observable_abs_error: Optional[float]
    error_in_std_errors: Optional[float]
    shot_noise_bound: float            # Hoeffding bound at shots_used

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        o, d = self.original, self.decomposed
        err = self.observable_abs_error
        z = self.error_in_std_errors
        rss = self.rss_delta_bytes
        return "\n".join([
            f"Qubits                          : {self.n_qubits}",
            "--- MPS (original, uncut) ---",
            f"  Max bond dimension            : {o.max_bond_dimension}",
            f"  Avg bond dimension            : {o.mean_avg_bond_dimension:.3f}",
            f"  Memory                        : {o.mean_memory_bytes / 1024:.2f} KiB",
            f"--- MPS (decomposed, {d.n_trajectories} trajectories) ---",
            f"  Max bond dimension (worst)    : {d.max_bond_dimension}",
            f"  Max bond dimension (mean±std) : {d.mean_max_bond_dimension:.2f} ± {d.std_max_bond_dimension:.2f}",
            f"  Avg bond dimension (mean)     : {d.mean_avg_bond_dimension:.3f}",
            f"  Memory (mean / peak)          : {d.mean_memory_bytes / 1024:.2f} / {d.peak_memory_bytes / 1024:.2f} KiB",
            f"  Bond reduction (max)          : {self.bond_reduction_max:.2f}x",
            f"  Memory reduction              : {self.memory_reduction:.2f}x",
            f"  RSS delta (trajectory loop)   : {'n/a' if rss is None else f'{rss / 2**20:.2f} MiB'}",
            "--- QPD ---",
            f"  Cut rotations                 : {self.num_cut_rotations}",
            f"  kappa                         : {self.kappa:.4f}",
            f"  Sampling overhead (kappa^2)   : {self.qpd_sampling_overhead:.4g}",
            f"  Optimal shots (eps, delta)    : {self.optimal_shots}",
            f"  Shots used                    : {self.shots_used}{'  (CAPPED)' if self.shots_capped else ''}",
            "--- Observable ---",
            f"  Reconstructed estimate        : {self.reconstructed_value:.6f} ± {self.reconstruction_std_error:.2e}",
            f"  Reference (original circuit)  : {self.reference_value}",
            f"  Abs. error                    : {'n/a' if err is None else f'{err:.3e}'}",
            f"  Error in std errors           : {'n/a' if z is None else f'{z:.2f}'}",
            f"  Hoeffding bound @ shots used  : {self.shot_noise_bound:.4g}",
        ])


# --------------------------------------------------------------------------- #
class MPSSimulationAnalyzer:
    """
    Parameters
    ----------
    circuit : QuantumCircuit
        The ORIGINAL (uncut) circuit.
    decomposer : MitaraiFujiDecomposition
        Configured decomposer. `analyze()` calls `decomposer.decompose(circuit)`.
    observable : SparsePauliOp | str | None
        Z/I-only observable on all original qubits (default Z^{⊗n}).
    n_trajectories : int
        Number of single-shot runs used for MPS statistics of the decomposed circuit.
    shots : int | None
        Shots for reconstruction. None -> decomposer's optimal shots (capped).
    max_shots : int
        Cap on reconstruction shots (kappa^2 grows fast with the number of cuts).
    max_bond_dimension, truncation_threshold :
        Aer MPS truncation settings (applied to both original and decomposed runs).
    reference : {"statevector", "mps_exact", None}
        How to compute the exact value of the ORIGINAL circuit.
    delta : failure probability for the Hoeffding bound.
    """

    def __init__(self, circuit: QuantumCircuit, decomposer,
                 observable: Optional[SparsePauliOp | str] = None,
                 n_trajectories: int = 100,
                 shots: Optional[int] = None,
                 max_shots: int = 200_000,
                 max_bond_dimension: Optional[int] = None,
                 truncation_threshold: float = 1e-16,
                 reference: Optional[str] = "statevector",
                 delta: float = 0.05,
                 seed: int = 1234):
        self.original = circuit.remove_final_measurements(inplace=False)
        self.n = self.original.num_qubits
        self.decomposer = decomposer

        if observable is None:
            observable = SparsePauliOp("Z" * self.n)
        elif isinstance(observable, str):
            observable = SparsePauliOp(observable)
        for p in observable.paulis:
            lab = p.to_label()
            if len(lab) != self.n or set(lab) - {"I", "Z"}:
                raise ValueError(f"Observable term '{lab}' must be a Z/I string of length {self.n}.")
        self.observable = observable

        self.n_trajectories = n_trajectories
        self.shots = shots
        self.max_shots = max_shots
        self.max_bond_dimension = max_bond_dimension
        self.truncation_threshold = truncation_threshold
        self.reference = reference
        self.delta = delta
        self.seed = seed

    # ------------------------------------------------------------ internals
    def _backend(self, seed: int) -> AerSimulator:
        return AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=self.max_bond_dimension,
            matrix_product_state_truncation_threshold=self.truncation_threshold,
            seed_simulator=seed,
        )

    @staticmethod
    def _bond_dims(mps, n_total: int) -> List[int]:
        gammas, lambdas = mps
        if len(lambdas) == n_total - 1:
            return [int(len(np.asarray(l))) for l in lambdas]
        dims = []
        for g in gammas[:-1]:
            part = g[0][0] if isinstance(g[0], (list, tuple)) else g[0]
            dims.append(int(np.asarray(part).shape[-1]))
        return dims

    @staticmethod
    def _nbytes(obj) -> int:
        if isinstance(obj, (list, tuple)):
            return sum(MPSSimulationAnalyzer._nbytes(o) for o in obj)
        return np.asarray(obj).nbytes

    @staticmethod
    def _pre_measurement_circuit(circuit: QuantumCircuit, n_out: int) -> QuantumCircuit:
        """Copy of `circuit` truncated before its trailing output measurements, with an MPS save.

        Trailing barriers/measurements into the first `n_out` clbits (the original output bits)
        are dropped; sel/beta measurements are earlier or inside control flow and are kept.
        """
        data = list(circuit.data)
        i = len(data)
        while i > 0:
            inst = data[i - 1]
            name = inst.operation.name
            if name == "barrier" or (
                name == "measure" and circuit.find_bit(inst.clbits[0]).index < n_out
            ):
                i -= 1
            else:
                break
        out = circuit.copy_empty_like()
        for inst in data[:i]:
            out.append(inst.operation, inst.qubits, inst.clbits)
        out.save_matrix_product_state(label="mps")
        return out

    def _aggregate(self, dims_list: List[List[int]], mem_list: List[int]) -> MPSStats:
        maxes = np.array([max(d) if d else 1 for d in dims_list], dtype=float)
        avgs = np.array([np.mean(d) if d else 1.0 for d in dims_list], dtype=float)
        profile = np.mean(np.array(dims_list, dtype=float), axis=0).tolist() if dims_list[0] else []
        return MPSStats(
            n_trajectories=len(dims_list),
            max_bond_dimension=int(maxes.max()),
            mean_max_bond_dimension=float(maxes.mean()),
            mean_avg_bond_dimension=float(avgs.mean()),
            mean_bond_profile=profile,
            mean_memory_bytes=float(np.mean(mem_list)),
            peak_memory_bytes=int(max(mem_list)),
            std_max_bond_dimension=float(maxes.std()),
        )
    
    def _observable_dict(self) -> dict:
        d = {}
        for p, c in zip(self.observable.paulis, self.observable.coeffs):
            d[p.to_label()] = d.get(p.to_label(), 0.0) + float(np.real(c))
        return d

    # ------------------------------------------------------ original circuit

    def _original_stats(self):
        qc = self.original.copy()
        qc.save_matrix_product_state(label="mps")
        qc.save_expectation_value(self.observable, list(range(self.n)), label="exp")
        data = self._backend(self.seed).run(qc, shots=1).result().data()
        dims = self._bond_dims(data["mps"], self.n)
        mem = self._nbytes(data["mps"])
        return self._aggregate([dims], [mem]), float(np.real(data["exp"]))

    def _reference_value(self, mps_exact_value: Optional[float]) -> Optional[float]:
        if self.reference == "statevector":
            return float(np.real(Statevector(self.original).expectation_value(self.observable)))
        if self.reference == "mps_exact":
            qc = self.original.copy()
            qc.save_expectation_value(self.observable, list(range(self.n)), label="exp")
            backend = AerSimulator(method="matrix_product_state",
                                   matrix_product_state_truncation_threshold=1e-16,
                                   seed_simulator=self.seed)
            return float(np.real(backend.run(qc, shots=1).result().data()["exp"]))
        return None


    # ------------------------------------------------------ decomposed circuit
    def _decomposed_mps_stats(self, decomposed: QuantumCircuit, n_out: int) -> MPSStats:
        probe = self._pre_measurement_circuit(decomposed, n_out)
        n_total = probe.num_qubits               # n (+1 ancilla if any cut)
        dims_list, mem_list = [], []
        for t in range(self.n_trajectories):
            data = self._backend(self.seed + 1 + t).run(probe, shots=1).result().data()
            dims = self._bond_dims(data["mps"], n_total)[: self.n - 1]   # drop ancilla bond
            dims_list.append(dims)
            mem_list.append(self._nbytes(data["mps"]))
        return self._aggregate(dims_list, mem_list)

    # ----------------------------------------------------------------- main
    def analyze(self) -> MPSReport:
        decomposed, optimal_shots = self.decomposer.decompose(self.original)
        plan = self.decomposer.plan
        kappa = plan.kappa

        # 1) original circuit
        orig_stats, mps_orig_value = self._original_stats()

        # 2) decomposed circuit: per-trajectory MPS statistics
        proc = psutil.Process() if psutil else None
        rss0 = proc.memory_info().rss if proc else None
        dec_stats = self._decomposed_mps_stats(decomposed, plan.num_output_clbits)
        rss1 = proc.memory_info().rss if proc else None

        # 3) decomposed circuit: full run with measurements, then reconstruct
        shots = self.shots if self.shots is not None else min(optimal_shots, self.max_shots)
        capped = self.shots is None and optimal_shots > self.max_shots
        result = self._backend(self.seed + 10_000).run(decomposed, shots=shots).result()
        rec = self.decomposer.reconstruct(result, observable=self._observable_dict(), circuit=decomposed)
        
        # 4) error vs ORIGINAL uncut circuit
        ref = self._reference_value(mps_orig_value)
        err = abs(rec.expectation - ref) if ref is not None else None
        z = (err / rec.std_error) if (err is not None and rec.std_error) else None

        obs_norm = float(np.sum(np.abs(self.observable.coeffs)))
        shot_bound = kappa * obs_norm * math.sqrt(2 * math.log(2 / self.delta) / shots)

        return MPSReport(
            n_qubits=self.n,
            original=orig_stats,
            decomposed=dec_stats,
            bond_reduction_max=orig_stats.max_bond_dimension / max(dec_stats.max_bond_dimension, 1),
            memory_reduction=orig_stats.mean_memory_bytes / max(dec_stats.mean_memory_bytes, 1),
            rss_delta_bytes=None if rss0 is None else rss1 - rss0,
            kappa=kappa,
            qpd_sampling_overhead=plan.sampling_overhead,
            optimal_shots=optimal_shots,
            shots_used=shots,
            shots_capped=capped,
            num_cut_rotations=len(plan.rotations),
            reconstructed_value=rec.expectation,
            reconstruction_std_error=rec.std_error,
            reference_value=ref,
            observable_abs_error=err,
            error_in_std_errors=z,
            shot_noise_bound=shot_bound,
        )


# --------------------------------------------------------------------------- #
# usage
# --------------------------------------------------------------------------- #
# decomposer = MitaraiFujiDecomposition(decom_percent=100, epsilon=0.05)
# analyser   = MPSSimulationAnalyzer(qc, decomposer, observable="ZZZZ",
#                                    n_trajectories=100, max_shots=100_000)
# report = analyser.analyze()
# print(report.summary())