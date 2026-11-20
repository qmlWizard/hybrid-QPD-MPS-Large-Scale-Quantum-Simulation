"""
Mitarai-Fujii "virtual" two-qubit gates for Qiskit 2.x
(K. Mitarai and K. Fujii, arXiv:1909.07534).

How a *single* circuit can hold a quasi-probability decomposition
------------------------------------------------------------------
Every factor exp(i*theta*A (x) A) of a two-qubit gate is replaced by a
randomly chosen branch (probability |c_i| / kappa, kappa = 1 + 2|sin 2theta|):

    branch 0 : I (x) I                         c = cos^2(theta)
    branch 1 : A (x) A                         c = sin^2(theta)
    branch 2 : M_A (x) exp(+i pi/4 A)          c = +cos(theta) sin(theta)
    branch 3 : M_A (x) exp(-i pi/4 A)          c = -cos(theta) sin(theta)
    branch 4 : exp(+i pi/4 A) (x) M_A          c = +cos(theta) sin(theta)
    branch 5 : exp(-i pi/4 A) (x) M_A          c = -cos(theta) sin(theta)

The branch is drawn *inside the circuit* by a quantum coin (one reusable
ancilla qubit ``vg_rng`` + reset + mid-circuit measurement) and applied with
``switch_case``.  All draws, and the projective-measurement outcomes, are
recorded in extra classical registers.  Each shot therefore yields one Monte-Carlo
sample with weight  kappa_total * prod(sign_i) * prod(beta_k),  and
``reconstruct`` re-weights the shots classically.

Requirements: a backend / simulator with dynamic-circuit support
(reset, mid-circuit measurement, switch_case) and one extra qubit.
No entanglement is created between the two qubits of a cut gate.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer, TwoQubitWeylDecomposition
from qiskit.transpiler import CouplingMap

__all__ = ["MitaraiFujiDecomposition", "DecompositionPlan", "ReconstructionResult"]

_META_KEY = "mitarai_fujii"
_NUM_FLIPS = 5          # stick-breaking coin flips used to sample 1 of 6 branches
_ANGLE_TOL = 1e-9
_PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _rotation_kappa(theta: float) -> float:
    """1-norm of the quasi-probability decomposition of exp(i*theta*A(x)A)."""
    return 1.0 + 2.0 * abs(math.sin(2.0 * theta))


def _branch_coeffs(theta: float) -> list[float]:
    c, s = math.cos(theta), math.sin(theta)
    return [c * c, s * s, c * s, -c * s, c * s, -c * s]


def _branch_values() -> dict[int, tuple[int, ...]]:
    """Register value -> branch. Branch = index of the first flip that came up 1."""
    table: dict[int, list[int]] = {i: [] for i in range(6)}
    for v in range(2 ** _NUM_FLIPS):
        table[((v & -v).bit_length() - 1) if v else 5].append(v)
    return {k: tuple(v) for k, v in table.items()}


def _first_set_branch(v: int) -> int:
    return ((v & -v).bit_length() - 1) if v else 5


@dataclass
class _GateCut:
    """U = e^{i phase} (K1l(x)K1r) exp(i(aXX+bYY+cZZ)) (K2l(x)K2r);  l -> qubit 1, r -> qubit 0."""
    rotations: list[tuple[str, float]]
    k1l: np.ndarray
    k1r: np.ndarray
    k2l: np.ndarray
    k2r: np.ndarray
    phase: float

    @property
    def log_kappa(self) -> float:
        return sum(math.log(_rotation_kappa(t)) for _, t in self.rotations)


def _weyl_cut(op: Gate) -> Optional[_GateCut]:
    """KAK-decompose a two-qubit gate; None if not numeric or purely local."""
    try:
        unitary = Operator(op).data
    except Exception:  # unbound parameters, opaque gate, ...
        return None
    w = TwoQubitWeylDecomposition(unitary, fidelity=None)
    rots = [(p, float(t)) for p, t in zip("XYZ", (w.a, w.b, w.c)) if abs(t) > _ANGLE_TOL]
    if not rots:
        return None
    # Defensive check of the (version dependent) KAK conventions.
    core = np.eye(4, dtype=complex)
    for p, t in rots:
        core = core @ (math.cos(t) * np.eye(4) + 1j * math.sin(t) * np.kron(_PAULI[p], _PAULI[p]))
    rec = np.exp(1j * w.global_phase) * np.kron(w.K1l, w.K1r) @ core @ np.kron(w.K2l, w.K2r)
    if not np.allclose(rec, unitary, atol=1e-7):
        raise RuntimeError("Unexpected KAK convention from TwoQubitWeylDecomposition.")
    return _GateCut(rots, w.K1l, w.K1r, w.K2l, w.K2r, float(w.global_phase))


def _emit_1q(qc: QuantumCircuit, qubit, mat: np.ndarray) -> None:
    if np.allclose(mat, mat[0, 0] * np.eye(2), atol=1e-10):
        qc.global_phase += float(np.angle(mat[0, 0]))
        return
    qc.compose(OneQubitEulerDecomposer("U")(mat), [qubit], inplace=True)


def _to_z_basis(qc, q, pauli):
    if pauli == "X":
        qc.h(q)
    elif pauli == "Y":
        qc.sdg(q)
        qc.h(q)


def _from_z_basis(qc, q, pauli):
    if pauli == "X":
        qc.h(q)
    elif pauli == "Y":
        qc.h(q)
        qc.s(q)


def _measure_pauli(qc, q, pauli, clbit):
    """Non-destructive projective measurement of `pauli` (bit 0 <-> eigenvalue +1)."""
    _to_z_basis(qc, q, pauli)
    qc.measure(q, clbit)
    _from_z_basis(qc, q, pauli)


def _rot(qc, q, pauli, sign):
    """Apply exp(i * sign * pi/4 * P) = R_P(-sign*pi/2)."""
    getattr(qc, "r" + pauli.lower())(-sign * math.pi / 2, q)


# --------------------------------------------------------------------------- #
# public data containers
# --------------------------------------------------------------------------- #
@dataclass
class DecompositionPlan:
    """Serializable record needed to post-process shots (also stored in circuit.metadata)."""
    num_clbits: int = 0
    num_output_clbits: int = 0
    creg_names: list[str] = field(default_factory=list)
    selected: list[int] = field(default_factory=list)      # instruction indices that were cut
    num_candidates: int = 0
    log_kappa: float = 0.0
    rotations: list[dict] = field(default_factory=list)    # {"sel":[5 clbits],"beta":clbit,"signs":[6]}

    @property
    def kappa(self) -> float:
        return math.exp(self.log_kappa) if self.log_kappa < 700 else float("inf")

    @property
    def sampling_overhead(self) -> float:
        """kappa^2 : multiplicative increase in shots w.r.t. the uncut circuit."""
        return self.kappa ** 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DecompositionPlan":
        return cls(**dict(d))


@dataclass
class ReconstructionResult:
    expectation: Optional[float]
    std_error: Optional[float]
    quasi_distribution: dict[str, float]   # keyed by the ORIGINAL clbits (clbit 0 rightmost); may be negative
    shots: int
    kappa: float


# --------------------------------------------------------------------------- #
# main class
# --------------------------------------------------------------------------- #
class MitaraiFujiDecomposition:
    """
    Parameters
    ----------
    decom_percent : percentage (0-100) of the *decomposable* two-qubit gates to cut.
                    100 -> every two-qubit gate is replaced by its virtual version.
    epsilon, delta : target accuracy / failure probability used by `get_optimal_shots`.
    coupling_map, initial_layout : optional. If given, gates acting on far-apart
                    physical qubits (SWAP cost) are preferred when choosing what to cut.
    strategy : "optimize" (default), "cheapest" (smallest kappa first) or "random".
    weights : dict with keys "depth", "routing", "kappa" (objective weights, default 1 each).
    durations : (1q duration, 2q duration) used by the depth model.
    max_search_evals : budget of objective evaluations for the local search.
    """

    def __init__(
        self,
        decom_percent: float = 100.0,
        epsilon: float = 0.01,
        delta: float = 0.05,
        *,
        coupling_map: Optional[CouplingMap] = None,
        initial_layout: Optional[Sequence[int]] = None,
        strategy: str = "optimize",
        weights: Optional[Mapping[str, float]] = None,
        durations: tuple[float, float] = (1.0, 10.0),
        max_search_evals: int = 2000,
        seed: Optional[int] = None,
    ):
        if not 0.0 <= decom_percent <= 100.0:
            raise ValueError("decom_percent must lie in [0, 100].")
        if not (0 < epsilon < 1 and 0 < delta < 1):
            raise ValueError("epsilon and delta must lie in (0, 1).")
        if strategy not in ("optimize", "cheapest", "random"):
            raise ValueError("strategy must be 'optimize', 'cheapest' or 'random'.")
        self.decom_percent = float(decom_percent)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.coupling_map = coupling_map
        self.initial_layout = list(initial_layout) if initial_layout is not None else None
        self.strategy = strategy
        w = {"depth": 1.0, "routing": 1.0, "kappa": 1.0}
        w.update(weights or {})
        self.weights = w
        self.durations = durations
        self.max_search_evals = max_search_evals
        self._rng = np.random.default_rng(seed)
        self.plan: Optional[DecompositionPlan] = None

    # ------------------------------------------------------------------ #
    # pre-processing
    # ------------------------------------------------------------------ #
    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Unroll every gate acting on >2 qubits so only 1q/2q gates remain.

        Native 2q gates (cx, cz, rzz, swap, ...) are kept as they are, because each is
        handled exactly through its KAK form. Control-flow bodies are not inspected.
        """
        qc = circuit
        for _ in range(32):
            big = {
                i.operation.name for i in qc.data
                if i.operation.num_qubits > 2
                and i.operation.name != "barrier"
                and not isinstance(i.operation, ControlFlowOp)
            }
            if not big:
                return qc.copy() if qc is circuit else qc
            qc = qc.decompose(gates_to_decompose=sorted(big))
        raise ValueError("Could not unroll all >2-qubit gates (opaque gate without definition?).")

    # ------------------------------------------------------------------ #
    # gate selection  (the optimisation step)
    # ------------------------------------------------------------------ #
    def _find_candidates(self, circuit: QuantumCircuit) -> dict[int, _GateCut]:
        cands: dict[int, _GateCut] = {}
        cache: dict[Any, Optional[_GateCut]] = {}
        for idx, inst in enumerate(circuit.data):
            op = inst.operation
            if isinstance(op, ControlFlowOp) or not isinstance(op, Gate) or op.num_qubits != 2:
                continue
            try:
                key = (op.name, tuple(float(p) for p in op.params))
            except (TypeError, ValueError):
                key = None
            if key is not None and key in cache:
                cut = cache[key]
            else:
                cut = _weyl_cut(op)
                if key is not None:
                    cache[key] = cut
            if cut is not None:
                cands[idx] = cut
        return cands

    def _num_to_cut(self, n_candidates: int) -> int:
        return min(n_candidates, int(math.floor(self.decom_percent / 100.0 * n_candidates + 0.5)))

    def select_gates_to_decompose(
        self, circuit: QuantumCircuit, candidates: Optional[dict[int, _GateCut]] = None
    ) -> list[int]:
        """Return the instruction indices (in ``circuit.data``) of the gates to cut.

        Chooses exactly K = round(decom_percent% * #candidates) gates maximising

            F(S) = w_depth   * (T(empty) - T(S)) / T(empty)      relative critical-path reduction
                 + w_routing * sum_{g in S} swaps_g / sum_all    SWAPs avoided (needs coupling_map)
                 - w_kappa   * sum_{g in S} log(kappa_g) / sum_all  sampling-cost penalty

        Method: rank gates by their stand-alone F({g}), take the top K, then improve with a
        1-exchange local search on the true (non-additive) F. When w_depth == 0 the objective is
        additive and the top-K ranking is already exactly optimal.
        """
        cands = self._find_candidates(circuit) if candidates is None else candidates
        ids = sorted(cands)
        k = self._num_to_cut(len(ids))
        if k == 0:
            return []
        if k == len(ids):
            return ids

        if self.strategy == "cheapest":
            return sorted(sorted(ids, key=lambda i: (cands[i].log_kappa, i))[:k])
        if self.strategy == "random":
            return sorted(int(i) for i in self._rng.choice(ids, size=k, replace=False))

        # ---- strategy == "optimize" ----
        nq = circuit.num_qubits
        d1, d2 = self.durations
        pos = {idx: p for p, idx in enumerate(ids)}
        ops = []  # (qubit indices, duration, candidate position or -1, local duration if cut)
        for idx, inst in enumerate(circuit.data):
            qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
            name = inst.operation.name
            if name == "barrier":
                dur = 0.0
            elif len(qs) == 1 or name == "measure":
                dur = d1
            else:
                dur = d2
            cp = pos.get(idx, -1)
            cut_dur = 2.0 * d1 * len(cands[idx].rotations) if cp >= 0 else 0.0
            ops.append((qs, dur, cp, cut_dur))

        def makespan(sel: set) -> float:
            t = [0.0] * nq
            for qs, dur, cp, cut_dur in ops:
                if cp >= 0 and cp in sel:
                    for q in qs:
                        t[q] += cut_dur
                elif len(qs) == 1:
                    t[qs[0]] += dur
                else:
                    m = max(t[q] for q in qs) + dur
                    for q in qs:
                        t[q] = m
            return max(t) if t else 0.0

        logk = [cands[i].log_kappa for i in ids]
        route = [0.0] * len(ids)
        if self.coupling_map is not None:
            layout = self.initial_layout or list(range(nq))
            if self.coupling_map.size() < nq:
                raise ValueError("coupling_map has fewer qubits than the circuit.")
            for p, idx in enumerate(ids):
                a, b = (layout[circuit.find_bit(q).index] for q in circuit.data[idx].qubits)
                route[p] = max(self.coupling_map.distance(a, b) - 1, 0)
        t0 = makespan(set())
        logk_tot, route_tot = sum(logk) or 1.0, sum(route) or 1.0
        wd, wr, wk = self.weights["depth"], self.weights["routing"], self.weights["kappa"]

        def objective(sel: set) -> float:
            depth_gain = (t0 - makespan(sel)) / t0 if (wd and t0 > 0) else 0.0
            return (wd * depth_gain
                    + wr * sum(route[p] for p in sel) / route_tot
                    - wk * sum(logk[p] for p in sel) / logk_tot)

        standalone = [objective({p}) for p in range(len(ids))]
        order = sorted(range(len(ids)), key=lambda p: (-standalone[p], p))
        current, best = set(order[:k]), None
        if wd:  # non-additive -> local search
            best, evals, m = objective(current), 0, 8
            improved = True
            while improved and evals < self.max_search_evals:
                improved = False
                outs = sorted(current, key=lambda p: standalone[p])[:m]
                ins = [p for p in order if p not in current][:m]
                for o in outs:
                    for i in ins:
                        trial = (current - {o}) | {i}
                        val = objective(trial)
                        evals += 1
                        if val > best + 1e-12:
                            current, best, improved = trial, val, True
                            break
                    if improved:
                        break
        return sorted(ids[p] for p in current)

    # ------------------------------------------------------------------ #
    # circuit construction
    # ------------------------------------------------------------------ #
    def decompose(self, circuit: QuantumCircuit) -> tuple[QuantumCircuit, int]:
        """Return ``(virtual_circuit, optimal_shots)``.

        A single circuit is returned (no sub-circuits). It has one extra qubit ``vg_rng``
        (randomness for branch sampling) and extra classical registers ``vg{k}_sel`` /
        ``vg{k}_m`` per cut rotation. Circuits without classical bits get ``measure_all()``.
        """
        base = self.transpile(circuit)
        if base.num_clbits == 0:
            base.measure_all()
        if any(not base.find_bit(c).registers for c in base.clbits):
            raise ValueError("All classical bits must belong to a ClassicalRegister.")

        cands = self._find_candidates(base)
        selected = self.select_gates_to_decompose(base, cands)
        plan = DecompositionPlan(
            num_output_clbits=base.num_clbits, selected=selected, num_candidates=len(cands)
        )

        if not selected:
            out = base
        else:
            out = base.copy_empty_like(name=f"{base.name}_virtual")
            rng_reg = QuantumRegister(1, "vg_rng")
            out.add_register(rng_reg)
            chosen = set(selected)
            for idx, inst in enumerate(base.data):
                if idx in chosen:
                    self._emit_gate(out, inst.qubits, cands[idx], rng_reg[0], plan)
                else:
                    out.append(inst.operation, inst.qubits, inst.clbits)

        plan.num_clbits = out.num_clbits
        plan.creg_names = [r.name for r in out.cregs]
        self.plan = plan
        out.metadata = {**(base.metadata or {}), _META_KEY: plan.to_dict()}
        return out, self.get_optimal_shots(out)

    def _emit_gate(self, out, qargs, cut: _GateCut, rng, plan: DecompositionPlan) -> None:
        q0, q1 = qargs                       # qiskit: 'r' factors act on q0, 'l' factors on q1
        _emit_1q(out, q0, cut.k2r)
        _emit_1q(out, q1, cut.k2l)
        for pauli, theta in cut.rotations:
            self._emit_rotation(out, q0, q1, pauli, theta, rng, plan)
        _emit_1q(out, q0, cut.k1r)
        _emit_1q(out, q1, cut.k1l)
        out.global_phase += cut.phase

    def _coefficients(self, theta: float) -> list[float]:
        """Quasi-probability coefficients [c1..c6] of exp(i*theta*A1(x)A2) (paper Eq. B2).

        Their absolute sum is kappa = 1 + 2|sin(2 theta)|  (3 for CX/CZ -> 9x more shots).
        """
        return _branch_coeffs(theta)

    def _emit_rotation(self, out, qa, qb, pauli, theta, rng, plan) -> None:
        k = len(plan.rotations)
        coeffs = self._coefficients(theta)
        kappa = sum(abs(c) for c in coeffs)
        probs = [abs(c) / kappa for c in coeffs]
        signs = [1 if c >= 0 else -1 for c in coeffs]

        sel = ClassicalRegister(_NUM_FLIPS, f"vg{k}_sel")
        beta = ClassicalRegister(1, f"vg{k}_m")
        out.add_register(sel, beta)

        # 1) sample the branch: stick-breaking coin flips on the ancilla
        remaining = 1.0
        for i in range(_NUM_FLIPS):
            q = probs[i] / remaining if remaining > 1e-15 else 0.0
            q = min(max(q, 0.0), 1.0)
            remaining -= probs[i]
            if q <= 0.0:
                continue                     # bit stays 0
            out.reset(rng)
            out.ry(2.0 * math.asin(math.sqrt(q)), rng)
            out.measure(rng, sel[i])

        # 2) apply the sampled branch (branch 0 = identity: no case)
        vals = _branch_values()
        with out.switch(sel) as case:
            with case(*vals[1]):
                getattr(out, pauli.lower())(qa)
                getattr(out, pauli.lower())(qb)
            with case(*vals[2]):
                _measure_pauli(out, qa, pauli, beta[0])
                _rot(out, qb, pauli, +1)
            with case(*vals[3]):
                _measure_pauli(out, qa, pauli, beta[0])
                _rot(out, qb, pauli, -1)
            with case(*vals[4]):
                _rot(out, qa, pauli, +1)
                _measure_pauli(out, qb, pauli, beta[0])
            with case(*vals[5]):
                _rot(out, qa, pauli, -1)
                _measure_pauli(out, qb, pauli, beta[0])

        plan.rotations.append({
            "sel": [out.find_bit(sel[i]).index for i in range(_NUM_FLIPS)],
            "beta": out.find_bit(beta[0]).index,
            "signs": signs,
        })
        plan.log_kappa += math.log(kappa)

    # ------------------------------------------------------------------ #
    # shots
    # ------------------------------------------------------------------ #
    def _log_kappa(self, circuit: Optional[QuantumCircuit]) -> float:
        if circuit is None:
            if self.plan is None:
                raise ValueError("Call decompose() first or pass a decomposed circuit.")
            return self.plan.log_kappa
        meta = (circuit.metadata or {}).get(_META_KEY)
        if meta is None:
            raise ValueError("Circuit was not produced by MitaraiFujiDecomposition.decompose().")
        return float(meta["log_kappa"])

    def get_optimal_shots(
        self, circuit: Optional[QuantumCircuit] = None,
        epsilon: Optional[float] = None, delta: Optional[float] = None,
    ) -> int:
        """Smallest N s.t. |estimate - truth| <= epsilon w.p. >= 1 - delta for any observable
        with |O| <= 1 (Hoeffding on samples bounded by kappa):

            N = 2 * kappa^2 * ln(2/delta) / epsilon^2

        (The paper writes ln(1/(2 delta)); ln(2/delta) is the standard two-sided form.)
        """
        eps = self.epsilon if epsilon is None else epsilon
        dlt = self.delta if delta is None else delta
        log_n = math.log(2.0 * math.log(2.0 / dlt) / eps ** 2) + 2.0 * self._log_kappa(circuit)
        if log_n > 700:
            print("Required number of shots is astronomically large (>1e300).")
            log_n = 700  # cap to avoid overflow in math.exp()
        n = math.ceil(math.exp(log_n))
        if n > 1e12:
            warnings.warn(f"Optimal shots = {n:.3e}; consider cutting fewer gates.", RuntimeWarning)
        return int(n)

    # ------------------------------------------------------------------ #
    # post-processing
    # ------------------------------------------------------------------ #
    def _full_counts(self, results, plan: DecompositionPlan) -> dict[str, int]:
        """Normalise supported result types to {full bitstring (no spaces, clbit 0 rightmost): count}."""
        if isinstance(results, Mapping):
            counts = results
        elif hasattr(results, "get_counts") and hasattr(results, "results"):   # qiskit.result.Result
            counts = results.get_counts()
            counts = counts[0] if isinstance(counts, list) else counts
        else:                                                                  # SamplerV2 output
            pub = results[0] if hasattr(results, "__getitem__") and not hasattr(results, "data") else results
            data = pub.data
            per_reg = [getattr(data, name).get_bitstrings() for name in plan.creg_names]
            counts = Counter("".join(t) for t in zip(*reversed(per_reg)))
        merged: Counter = Counter()
        for key, n in counts.items():
            key = key.replace(" ", "")
            if len(key) != plan.num_clbits:
                raise ValueError(
                    f"Bitstring length {len(key)} != {plan.num_clbits}: results must contain ALL "
                    "classical registers of the decomposed circuit.")
            merged[key] += int(n)
        return dict(merged)

    @staticmethod
    def _build_observable(obs, n_out: int) -> Optional[Callable[[str], float]]:
        if obs is None:
            return None
        if hasattr(obs, "paulis") and hasattr(obs, "coeffs"):     # SparsePauliOp (must come BEFORE callable check)
            merged = {}
            for p, c in zip(obs.paulis, obs.coeffs):
                merged[p.to_label()] = merged.get(p.to_label(), 0.0) + float(np.real(c))
            obs = merged
        elif callable(obs):
            return obs
        if isinstance(obs, str):
            obs = {obs: 1.0}
        terms = []
        for label, coeff in obs.items():
            label = label.upper()
            if len(label) != n_out or set(label) - {"I", "Z"}:
                raise ValueError(f"Observable '{label}' must be a Z/I string of length {n_out}.")
            terms.append(([i for i, ch in enumerate(label) if ch == "Z"], float(coeff)))

        def f(y: str) -> float:
            tot = 0.0
            for mask, c in terms:
                tot += -c if sum(y[i] == "1" for i in mask) & 1 else c
            return tot
        return f

    def reconstruct(self, results, observable=None, circuit: Optional[QuantumCircuit] = None
                    ) -> ReconstructionResult:
        """Re-weight shots into an unbiased estimate of the *uncut* circuit.

        results    : SamplerV2 PrimitiveResult / pub result, qiskit.result.Result, or a counts dict
                     that includes ALL classical registers of the decomposed circuit.
        observable : None | 'ZIZ...' string | {label: coeff} | SparsePauliOp (Z/I only) |
                     callable(bitstring)->float.  Labels are aligned with the ORIGINAL clbits
                     (same orientation as Qiskit counts).
        circuit    : decomposed circuit (optional; defaults to the last one from decompose()).
        """
        if circuit is not None:
            plan = DecompositionPlan.from_dict(circuit.metadata[_META_KEY])
        elif self.plan is not None:
            plan = self.plan
        else:
            raise ValueError("Call decompose() first or pass the decomposed circuit.")

        counts = self._full_counts(results, plan)
        shots = sum(counts.values())
        n_tot, n_out = plan.num_clbits, plan.num_output_clbits
        f = self._build_observable(observable, n_out)
        kappa = plan.kappa
        rot = [(r["sel"], r["beta"], r["signs"]) for r in plan.rotations]

        quasi: dict[str, float] = {}
        s1 = s2 = 0.0
        for key, n in counts.items():
            sgn = 1
            for sel, beta, signs in rot:
                v = 0
                for i, c in enumerate(sel):
                    if key[n_tot - 1 - c] == "1":
                        v |= 1 << i
                b = _first_set_branch(v)
                sgn *= signs[b]
                if b >= 2 and key[n_tot - 1 - beta] == "1":
                    sgn = -sgn
            y = key[n_tot - n_out:]
            quasi[y] = quasi.get(y, 0.0) + kappa * sgn * n / shots
            if f is not None:
                x = kappa * sgn * f(y)
                s1 += n * x
                s2 += n * x * x

        exp_val = std_err = None
        if f is not None:
            exp_val = s1 / shots
            var = max(s2 / shots - exp_val ** 2, 0.0) * shots / max(shots - 1, 1)
            std_err = math.sqrt(var / shots)
        return ReconstructionResult(exp_val, std_err, quasi, shots, kappa)