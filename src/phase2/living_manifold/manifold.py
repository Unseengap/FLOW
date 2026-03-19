"""Living Manifold — the dynamic geometric knowledge space.

M(t) = (M₀, φ(t), ρ(t), κ(t))

This class is the central data structure of Phase 2.  It wraps the static
SeedManifold M₀ produced by Phase 1, adds mutable state (deformations,
density, curvature), and exposes both READ and WRITE operation sets.

READ  operations are called by the Flow Engine and Resonance Layer.
WRITE operations are called by the Annealing Engine and Contrast Engine.

Locality guarantee (hard constraint):
  Any deformation at P affects only points within locality_radius(P).
  effect(Q) → 0 as distance(P, Q) → ∞.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

# Phase 1 types
from src.phase1.seed_geometry.manifold import ManifoldPoint, SeedManifold
from src.phase1.seed_geometry.composer import FiberBundleComposer

from .deformation import LocalDeformation
from .geodesic import GeodesicComputer
from .regions import RegionClassifier, RegionType
from .state import DeformationField, DensityField, ManifoldState


class LivingManifold:
    """Dynamic Riemannian manifold data structure.

    Parameters
    ----------
    seed : SeedManifold
        The static M₀ produced by Phase 1.  Used as the initial state.
    density_radius : float
        The fixed radius used to count neighbours for density estimation.
    k_geodesic : int
        Number of nearest neighbours in the geodesic kNN graph.
    """

    DIM = 104  # total bundle dimension

    def __init__(
        self,
        seed: SeedManifold,
        density_radius: float = 3.0,
        k_geodesic: int = 8,
    ) -> None:
        self._seed = seed
        self._density_radius = density_radius
        self._classifier = RegionClassifier()
        self._deformer = LocalDeformation(cutoff_sigma=3.0)
        self._geodesic = GeodesicComputer(k_neighbours=k_geodesic)
        self._state = ManifoldState()
        self._build_start = time.monotonic()

        # Primary store: label → current position vector (including deformation)
        self._points: Dict[str, np.ndarray] = {}

        # All ManifoldPoint metadata (label, origin, …)
        self._metadata: Dict[str, ManifoldPoint] = {}

        # cKDTree for fast nearest-neighbour lookup (lazy rebuild)
        self._kdtree: Optional[cKDTree] = None
        self._kdtree_labels: List[str] = []
        self._kdtree_dirty: bool = True
        self._kdtree_writes_since_rebuild: int = 0
        self._kdtree_rebuild_interval: int = 50  # rebuild at most every N writes

        # Maximum number of neighbors affected per deformation (scalability cap)
        self._max_deform_neighbors: int = 64

        # Composer reused from seed (already initialised with all 4 geometries)
        self._composer = seed.composer

        # Ingest seed points
        self._load_seed(seed)

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def _load_seed(self, seed: SeedManifold) -> None:
        for mp in seed.seed_points:
            self._register_point(mp)
        # Compute initial densities
        for label in list(self._points.keys()):
            self._recompute_density(label)

    def _register_point(self, mp: ManifoldPoint) -> None:
        """Add a ManifoldPoint to all internal stores."""
        label = mp.label
        self._points[label] = mp.vector.copy()
        self._metadata[label] = mp
        self._state.deformation.register(label, self.DIM)
        self._geodesic.add_point(label, mp.vector)
        self._kdtree_dirty = True

    # ------------------------------------------------------------------ #
    # KD-tree (lazy)                                                       #
    # ------------------------------------------------------------------ #

    def _ensure_kdtree(self, force: bool = False) -> None:
        """Rebuild the cKDTree if stale.

        The tree is considered stale when ``_kdtree_dirty`` is True **and**
        either *force* is True or enough writes have accumulated since the
        last rebuild (controlled by ``_kdtree_rebuild_interval``).  This
        avoids O(n) rebuilds after every single ``place()``/``deform_local()``
        call, which was the dominant bottleneck at 50K+ manifold points.
        """
        if not self._kdtree_dirty:
            return
        # First build is always forced; after that, throttle rebuilds
        if (self._kdtree is not None and not force
                and self._kdtree_writes_since_rebuild < self._kdtree_rebuild_interval):
            return
        labels = list(self._points.keys())
        if not labels:
            self._kdtree = None
            self._kdtree_labels = []
            self._kdtree_dirty = False
            return
        matrix = np.stack([self._points[l] for l in labels])
        self._kdtree = cKDTree(matrix)
        self._kdtree_labels = labels
        self._kdtree_dirty = False
        self._kdtree_writes_since_rebuild = 0

    # ------------------------------------------------------------------ #
    # READ operations                                                      #
    # ------------------------------------------------------------------ #

    def position(self, concept: str) -> np.ndarray:
        """Current position of *concept* in M(t).

        Returns the seed position plus any accumulated deformation.

        Raises
        ------
        KeyError if the concept is not on the manifold.
        """
        if concept not in self._points:
            raise KeyError(f"Concept '{concept}' not found on manifold.")
        return self._points[concept].copy()

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Approximate geodesic distance between two position vectors.

        For the general case (unlabelled vectors) we use the bundle metric
        approximation from FiberBundleComposer.  This is cheaper than
        full Dijkstra for arbitrary vectors.
        """
        return float(self._composer.bundle_distance(p1, p2))

    def geodesic(self, label_a: str, label_b: str) -> List[np.ndarray]:
        """Approximate geodesic path as a list of position vectors.

        The path is computed via Dijkstra on the kNN graph and returned
        as an ordered sequence of 104D position vectors.
        """
        path_labels = self._geodesic.path(label_a, label_b)
        return [self._points[l] for l in path_labels if l in self._points]

    def geodesic_distance(self, label_a: str, label_b: str) -> float:
        """Graph-based geodesic distance between two labelled points."""
        return self._geodesic.distance(label_a, label_b)

    def curvature(self, p: np.ndarray) -> float:
        """Scalar curvature estimate at position *p*.

        We estimate curvature as the inverse mean distance to the k nearest
        neighbours — denser regions are more curved (tighter packing → higher
        curvature).
        """
        self._ensure_kdtree()
        if self._kdtree is None:
            return 0.0
        k = min(8, len(self._kdtree_labels) - 1)
        if k < 1:
            return 0.0
        dists, _ = self._kdtree.query(p, k=k + 1)
        dists = dists[1:]  # drop self (dist=0)
        mean_dist = float(np.mean(dists))
        return 1.0 / (mean_dist + 1e-8)

    def density(self, p: np.ndarray) -> float:
        """Normalised local density at position *p* in [0, 1].

        Counts how many manifold points lie within *density_radius* and
        normalises by the theoretical maximum (all points in the ball).
        """
        self._ensure_kdtree()
        if self._kdtree is None:
            return 0.0
        count = len(self._kdtree.query_ball_point(p, self._density_radius))
        total = max(len(self._points), 1)
        raw = count / total
        # Normalise to [0, 1] using a soft cap so that very dense regions
        # saturate at 1 but typical densities remain differentiated.
        return float(min(raw * 20.0, 1.0))

    def neighbors(self, p: np.ndarray, r: float) -> List[Tuple[str, np.ndarray]]:
        """All labelled points within radius *r* of position *p*.

        Returns
        -------
        List of (label, vector) tuples, sorted by distance ascending.
        """
        self._ensure_kdtree()
        if self._kdtree is None:
            return []
        idxs = self._kdtree.query_ball_point(p, r)
        results = []
        for i in idxs:
            label = self._kdtree_labels[i]
            vec = self._points[label]
            results.append((label, vec, float(np.linalg.norm(vec - p))))
        results.sort(key=lambda x: x[2])
        return [(l, v) for l, v, _ in results]

    def nearest(self, p: np.ndarray, k: int = 1) -> List[Tuple[str, np.ndarray]]:
        """Return the *k* nearest labelled points to *p*."""
        self._ensure_kdtree()
        if self._kdtree is None:
            return []
        k_actual = min(k, len(self._kdtree_labels))
        dists, idxs = self._kdtree.query(p, k=k_actual)
        if k_actual == 1:
            idxs = [idxs]
        return [
            (self._kdtree_labels[i], self._points[self._kdtree_labels[i]])
            for i in idxs
        ]

    def causal_direction(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Unit vector pointing from *p1* toward *p2* in the causal fiber.

        The causal fiber occupies dimensions 64–79 (indices) of the bundle.
        We return the unit difference in that sub-space, embedded back into
        the full 104D space.

        If the causal sub-vectors are identical, returns zero vector.
        """
        CAUSAL_SLICE = slice(64, 80)
        diff = p2[CAUSAL_SLICE] - p1[CAUSAL_SLICE]
        norm = float(np.linalg.norm(diff))
        result = np.zeros(self.DIM)
        if norm > 1e-12:
            result[CAUSAL_SLICE] = diff / norm
        return result

    def domain_of(self, p: np.ndarray) -> str:
        """Domain label for position *p* via similarity geometry."""
        BASE = slice(0, 64)
        return self._composer.sim.domain_of(p[BASE])

    def locality_radius(self, p: np.ndarray) -> float:
        """Deformation locality radius at *p* (denser → smaller)."""
        dens = self.density(p)
        return self._classifier.locality_radius(dens)

    def causal_ancestry(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """True if p1 is causally upstream of p2.

        Works directly on the causal-fiber slice (dims 64-79) of the raw
        position vectors returned by :meth:`position`.
        """
        CAUSAL = slice(64, 80)
        return self._composer.cau.is_causal_ancestor(p1[CAUSAL], p2[CAUSAL])

    def confidence(self, p: np.ndarray) -> float:
        """Confidence at *p* derived from local density."""
        return self.density(p)

    def region_type(self, p: np.ndarray) -> RegionType:
        """Qualitative region type at position *p*."""
        dens = self.density(p)
        return self._classifier.classify(dens)

    def logic_certainty(self, p: np.ndarray) -> float:
        """Logic certainty at *p* from the logical fiber (dims 80-87).

        Returns 1 - uncertainty_score: 1.0 = vertex (fully certain),
        0.0 = centroid (maximally uncertain).
        """
        LOGICAL = slice(80, 88)
        return 1.0 - self._composer.log.uncertainty_score(p[LOGICAL])

    # ------------------------------------------------------------------ #
    # WRITE operations                                                     #
    # ------------------------------------------------------------------ #

    def deform_local(self, label: str, delta: np.ndarray) -> int:
        """Apply a local deformation centred at point *label*.

        The deformation propagates with Gaussian falloff; only points
        within 3 * locality_radius(label) are moved.  Stiffer (denser)
        points resist movement.

        Returns the number of affected points.
        """
        if label not in self._points:
            raise KeyError(f"Cannot deform: '{label}' not on manifold.")

        centre_vec = self._points[label]
        dens = self._state.density.get(label)
        r = self._classifier.locality_radius(dens)

        # Pre-filter candidates using cKDTree range query — O(k_local)
        # instead of scanning all n points.
        cutoff = self._deformer.cutoff_sigma * r
        self._ensure_kdtree()
        candidate_labels = None
        if self._kdtree is not None and cutoff > 0:
            idxs = self._kdtree.query_ball_point(centre_vec, cutoff)
            # Cap the number of affected neighbours to prevent O(n)
            # degradation in dense manifold regions.  The centre point
            # is always included; remaining candidates are the closest.
            if len(idxs) > self._max_deform_neighbors:
                # Keep nearest neighbours only (by index proximity to
                # the kdtree data, which is spatially coherent)
                dists = np.linalg.norm(
                    self._kdtree.data[idxs] - centre_vec, axis=1
                )
                keep = np.argsort(dists)[: self._max_deform_neighbors]
                idxs = [idxs[k] for k in keep]
            candidate_labels = {self._kdtree_labels[i] for i in idxs}

        result = self._deformer.apply(
            centre_label=label,
            centre_vector=centre_vec,
            delta=delta,
            locality_radius=r,
            all_points=self._points,
            density_func=lambda l: self._state.density.get(l),
            candidate_labels=candidate_labels,
        )

        # Apply computed displacements
        for affected_label, displacement in result.affected:
            self._points[affected_label] = (
                self._points[affected_label] + displacement
            )
            self._state.deformation.accumulate(affected_label, displacement)
            self._geodesic.update_point(
                affected_label, self._points[affected_label]
            )

        self._kdtree_dirty = True
        self._kdtree_writes_since_rebuild += 1
        self._state.tick()
        return result.n_affected

    def place(self, concept: str, vector: np.ndarray, origin: str = "placed") -> ManifoldPoint:
        """Create a new concept at exact position *vector*.

        If *concept* already exists it is moved to the new position.

        Returns the new ManifoldPoint.
        """
        mp = ManifoldPoint(
            vector=vector.copy(),
            label=concept,
            origin=origin,
        )
        self._points[concept] = vector.copy()
        self._metadata[concept] = mp
        self._state.deformation.register(concept, self.DIM)
        self._geodesic.add_point(concept, vector)
        self._kdtree_dirty = True
        self._kdtree_writes_since_rebuild += 1
        self._recompute_density(concept)
        self._state.tick()
        return mp

    def place_fast(self, concept: str, vector: np.ndarray, origin: str = "placed") -> ManifoldPoint:
        """Fast placement for batch vocabulary loading.

        Same as ``place()`` but skips per-point density and curvature
        recomputation (those require kNN queries that are O(n) when the
        tree is stale).  Call ``flush_batch()`` after a batch to rebuild
        the tree and recompute densities in one pass.

        Returns the new ManifoldPoint.
        """
        mp = ManifoldPoint(
            vector=vector.copy(),
            label=concept,
            origin=origin,
        )
        self._points[concept] = vector.copy()
        self._metadata[concept] = mp
        self._state.deformation.register(concept, self.DIM)
        self._geodesic.add_point(concept, vector)
        self._kdtree_dirty = True
        self._kdtree_writes_since_rebuild += 1
        # Set a default density — will be recomputed in flush_batch()
        self._state.density.set(concept, 0.0)
        self._state.tick()
        return mp

    def flush_batch(self, labels: Optional[List[str]] = None) -> None:
        """Force a KDTree rebuild and recompute densities for *labels*.

        If *labels* is None, all points are refreshed.  Call this after a
        sequence of ``place_fast()`` calls to bring density / curvature
        estimates up to date.
        """
        self._ensure_kdtree(force=True)
        targets = labels if labels is not None else list(self._points.keys())
        for lbl in targets:
            if lbl in self._points:
                self._recompute_density(lbl)

    def force_rebuild_tree(self) -> None:
        """Force an immediate cKDTree rebuild.

        Useful after a large batch of writes when you need accurate
        nearest-neighbour queries immediately.
        """
        self._ensure_kdtree(force=True)

    def update_density(self, label: str) -> float:
        """Recompute density at *label* based on current neighbour counts.

        Returns the new density value.
        """
        if label not in self._points:
            raise KeyError(f"Cannot update density: '{label}' not on manifold.")
        return self._recompute_density(label)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _recompute_density(self, label: str) -> float:
        """Compute and store density for *label*."""
        vec = self._points[label]
        dens = self.density(vec)
        self._state.density.set(label, dens)
        self._state.set_curvature(label, self.curvature(vec))
        return dens

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        """Basic sanity checks on the manifold state.

        Returns True if all checks pass; raises AssertionError otherwise.
        """
        assert len(self._points) > 0, "Manifold has no points."
        for label, vec in self._points.items():
            assert vec.shape == (self.DIM,), (
                f"Point '{label}' has wrong dimension {vec.shape}."
            )
            assert np.all(np.isfinite(vec)), (
                f"Point '{label}' contains non-finite values."
            )
        return True

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n_points(self) -> int:
        """Total number of points on the manifold."""
        return len(self._points)

    @property
    def t(self) -> float:
        """Current manifold time."""
        return self._state.t

    @property
    def n_writes(self) -> int:
        """Total number of write operations committed so far."""
        return self._state.n_writes

    @property
    def labels(self) -> List[str]:
        """All labels currently on the manifold."""
        return list(self._points.keys())

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        elapsed = time.monotonic() - self._build_start
        # Region breakdown
        counts = {rt: 0 for rt in RegionType}
        for label in self._points:
            dens = self._state.density.get(label)
            rt = self._classifier.classify(dens)
            counts[rt] += 1

        lines = [
            "═══ Living Manifold M(t) ════════════════════════════════════",
            f"  Points          : {self.n_points}",
            f"  Dimension       : {self.DIM}",
            f"  Manifold time t : {self.t:.3f}",
            f"  Write ops       : {self.n_writes}",
            f"  Uptime          : {elapsed:.3f}s",
            "  Regions:",
            f"    Crystallized  : {counts[RegionType.CRYSTALLIZED]}",
            f"    Flexible      : {counts[RegionType.FLEXIBLE]}",
            f"    Unknown       : {counts[RegionType.UNKNOWN]}",
            "═════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"LivingManifold(n_points={self.n_points}, t={self.t:.2f}, writes={self.n_writes})"
