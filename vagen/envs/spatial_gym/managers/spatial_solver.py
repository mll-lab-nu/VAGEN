from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Union, Iterable
import random
import copy

# Import the relationship classes from the existing codebase
from ..core.relationship import (
    PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship,
    RelationTriple, CardinalBinsAllo, StandardDistanceBins
)
from ..utils.relationship_utils import relationship_applies, generate_points_for_relationship


@dataclass
class Variable:
    name: str
    domain: Optional[Set[Tuple[int, int]]] = None


@dataclass
class Constraint:
    var1_name: str
    var2_name: str
    relation: Union[PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship]
    orientation: Tuple[int, int] = (0, 1)
    
    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return False
        return (self.var1_name == other.var1_name and 
                self.var2_name == other.var2_name and
                self.relation == other.relation and
                self.orientation == other.orientation)
    
    def __hash__(self):
        return hash((self.var1_name, self.var2_name, self.relation, self.orientation))


class AC3Solver:
    def __init__(self, variables: Dict[str, Variable], constraints: List[Constraint]):
        self.variables = variables
        self.adjacency: Dict[str, Set[str]] = {name: set() for name in self.variables}
        # Group constraints by variable pairs (undirected)
        self.arc_constraints: Dict[frozenset, Set[Constraint]] = {}
        # Residue cache: (Xi, Xj) -> {vi: vj}
        self._residue: Dict[Tuple[str, str], Dict[Tuple[int, int], Tuple[int, int]]] = {}
        # Path-consistency cache: (A, pA, B, pB) -> bool
        self._pc_cache: Dict[Tuple[str, Tuple[int, int], str, Tuple[int, int]], bool] = {}
        
        for c in constraints:
            self.add_constraint(c)

    def add_constraint(self, constraint: Constraint) -> bool:
        """Add constraint to arc predicates. Returns True if arc is new or predicate changed."""
        var1, var2 = constraint.var1_name, constraint.var2_name
        # Use frozenset for undirected arc storage
        arc = frozenset({var1, var2})
        
        # Update adjacency
        self.adjacency.setdefault(var1, set()).add(var2)
        self.adjacency.setdefault(var2, set()).add(var1)
        
        # Add constraint to arc predicate
        old_constraints = self.arc_constraints.get(arc, set()).copy()
        self.arc_constraints.setdefault(arc, set()).add(constraint)
        
        # Return True if predicate changed (new constraint added)
        changed = old_constraints != self.arc_constraints[arc]
        if changed:
            # Clear caches related to this arc
            self._pc_cache.clear()
            self._residue.pop((var1, var2), None)
            self._residue.pop((var2, var1), None)
        return changed

    def copy(self) -> 'AC3Solver':
        new_variables = {
            name: Variable(name=name, domain=(set(var.domain) if var.domain is not None else None))
            for name, var in self.variables.items()
        }

        flat = {c for bucket in self.arc_constraints.values() for c in bucket}
        new_constraints = [copy.deepcopy(c) for c in flat]
        return AC3Solver(new_variables, new_constraints)

    def __deepcopy__(self):
        return self.copy()

    def propagate(self, changed_arcs: Optional[Set[Tuple[str, str]]] = None) -> bool:
        """AC-3 propagation from changed arcs or all arcs."""
        # invalidate caches
        self._pc_cache.clear()
        if changed_arcs is None:
            # All directed arcs
            work: Set[Tuple[str, str]] = {(a, b) for a, ns in self.adjacency.items() for b in ns}
        else:
            work = changed_arcs.copy()
                
        while work:
            var1_name, var2_name = work.pop()
            if self._revise(var1_name, var2_name):
                d = self.variables[var1_name].domain
                if d is not None and len(d) == 0:
                    raise ValueError(f"Domain is empty for {var1_name}")
                # Add all incoming arcs to var1 (except the one we just processed)
                for neighbor in self.adjacency[var1_name]:
                    if neighbor != var2_name:
                        work.add((neighbor, var1_name))
        return True

    def _revise(self, var1_name: str, var2_name: str) -> bool:
        revised = False
        if self.variables[var1_name].domain is None:
            return False
        var1_domain = self.variables[var1_name].domain.copy()
        if self.variables[var2_name].domain is None:
            return False
        for pos1 in var1_domain:
            if not self._has_support(var1_name, var2_name, pos1):
                self.variables[var1_name].domain.remove(pos1)
                revised = True
        return revised

    def _constraints_between(self, a: str, b: str) -> Set[Constraint]:
        """Get all constraints between variables a and b (undirected)."""
        arc = frozenset({a, b})
        return self.arc_constraints.get(arc, set())

    def _has_support(self, var1_name: str, var2_name: str, pos1: Tuple[int, int]) -> bool:
        """
        Check if pos1 from var1 has support in var2's domain.
        For each value pos2 in var2's domain, check if ALL constraints are satisfied (conjunction).
        """
        constraints = self._constraints_between(var1_name, var2_name)
        if not constraints:
            return True
            
        var2_domain = self.variables[var2_name].domain
        if var2_domain is None:
            return True  # cannot decide yet; defer pruning
        # Residue fast path
        arc_key = (var1_name, var2_name)
        cached_map = self._residue.get(arc_key)
        if cached_map is not None:
            cached_pos2 = cached_map.get(pos1)
            if cached_pos2 is not None and cached_pos2 in var2_domain:
                all_satisfied = True
                for constraint in constraints:
                    if constraint.var1_name == var1_name:
                        satisfied = relationship_applies(pos1, cached_pos2, constraint.relation, constraint.orientation)
                    else:
                        satisfied = relationship_applies(cached_pos2, pos1, constraint.relation, constraint.orientation)
                    if not satisfied:
                        all_satisfied = False
                        break
                if all_satisfied:
                    return True

        for pos2 in var2_domain:
            # Check if ALL constraints are satisfied for this pos2 (conjunction)
            all_satisfied = True
            for constraint in constraints:
                # Determine the correct order based on constraint definition
                if constraint.var1_name == var1_name:
                    # Constraint is defined as var1 -> var2, so check pos1 -> pos2
                    satisfied = relationship_applies(pos1, pos2, constraint.relation, constraint.orientation)
                else:
                    # Constraint is defined as var2 -> var1, so check pos2 -> pos1
                    satisfied = relationship_applies(pos2, pos1, constraint.relation, constraint.orientation)
                
                if not satisfied:
                    all_satisfied = False
                    break
            
            # If ALL constraints are satisfied for this pos2, then pos1 has support
            if all_satisfied:
                # Save residue
                self._residue.setdefault(arc_key, {})[pos1] = pos2
                return True
                
        # No value in var2's domain satisfies all constraints
        return False

    def _pc_key(self, a: str, pa: Tuple[int, int], b: str, pb: Tuple[int, int]) -> Tuple[str, Tuple[int, int], str, Tuple[int, int]]:
        # Symmetric cache key
        return (a, pa, b, pb) if (a, pa) <= (b, pb) else (b, pb, a, pa)

    def _pair_constraints_satisfied(self, a: str, pa: Tuple[int, int], b: str, pb: Tuple[int, int]) -> bool:
        for c in self._constraints_between(a, b):
            if c.var1_name == a:
                if not relationship_applies(pa, pb, c.relation, c.orientation):
                    return False
            else:
                if not relationship_applies(pb, pa, c.relation, c.orientation):
                    return False
        return True

    def is_pair_value_path_consistent(self, var1_name: str, pos1: Tuple[int, int],
                                      var2_name: str, pos2: Tuple[int, int]) -> bool:
        """Check path-consistency for (var1=pos1, var2=pos2).
        Require C_AB(pos1,pos2) and for every k != A,B, exists pk in Dk s.t.
        C_Ak(pos1,pk) and C_Bk(pos2,pk) (where present)."""
        key = self._pc_key(var1_name, pos1, var2_name, pos2)
        cached = self._pc_cache.get(key)
        if cached is not None:
            return cached

        # C_AB(pos1, pos2)
        if not self._pair_constraints_satisfied(var1_name, pos1, var2_name, pos2):
            self._pc_cache[key] = False
            return False

        # For all k
        for k in self.variables.keys():
            if k == var1_name or k == var2_name:
                continue
            if not self._constraints_between(var1_name, k) and not self._constraints_between(var2_name, k):
                continue
            domain_k = self.variables[k].domain
            assert domain_k is not None and len(domain_k) > 0, f"Domain is empty for {k}"
            supported = False
            for posk in domain_k:
                if self._pair_constraints_satisfied(var1_name, pos1, k, posk) and \
                   self._pair_constraints_satisfied(var2_name, pos2, k, posk):
                    supported = True
                    break
            if not supported:
                self._pc_cache[key] = False
                return False

        self._pc_cache[key] = True
        return True


class SpatialSolver:
    """Spatial constraint solver with AC-3 and simple metrics."""

    def __init__(self, all_object_names: List[str], grid_size: int):
        self.grid_size = int(grid_size)
        variables = {name: Variable(name=name, domain=None) for name in all_object_names}
        self.solver = AC3Solver(variables, [])

    def set_initial_position(self, name: str, position: Tuple[int, int]):
        if name not in self.solver.variables:
            self.solver.variables[name] = Variable(name=name, domain=None)
        self.solver.variables[name].domain = {tuple(position)}

    def add_observation(self, relation_triples: List[RelationTriple]) -> bool:
        """Add observations: seed domains cheaply, then solve all constraints.
        Returns True if all variables keep non-empty domains; False otherwise.
        """
        new_constraints: List[Constraint] = []
        # 1) seeding from pairwise constraints only
        g = int(self.grid_size)
        x_rng, y_rng = (-g, g), (-g, g)
        relation_triples.sort(key=lambda t: t.anchor != 'initial_pos')
        for t in relation_triples:
            cons = Constraint(t.subject, t.anchor, t.relation, t.orientation or (0, 1))
            new_constraints.append(cons)
            # seed subject via anchor for pairwise only
            if isinstance(t.relation, (PairwiseRelationship, PairwiseRelationshipDiscrete)):
                s_dom = self.solver.variables[t.subject].domain
                a_dom = self.solver.variables[t.anchor].domain
                # Initialized vs uninitialized: empty set means uninitialized for our solver; we use a sentinel None to mark uninitialized
                if s_dom is not None and len(s_dom) == 0:
                    # already unsatisfiable; cannot seed
                    pass
                if (s_dom is None) and isinstance(a_dom, set) and 1 <= len(a_dom) <= 10:
                    domain = set()
                    for anchor_pt in a_dom:
                        domain |= generate_points_for_relationship(anchor_pt, t.relation, x_rng, y_rng, t.orientation or (0, 1))
                    self.solver.variables[t.subject].domain = domain
            
        for t in relation_triples:
            # ensure domains exist only if uninitialized
            self._ensure_domain_initialized(t.subject)
            self._ensure_domain_initialized(t.anchor)

        # 2) install constraints and propagate from changed arcs only
        changed_arcs = set()
        for c in new_constraints:
            if self.solver.add_constraint(c):
                # Track arcs that need propagation (both directions)
                changed_arcs.add((c.var1_name, c.var2_name))
                changed_arcs.add((c.var2_name, c.var1_name))
        
        if changed_arcs:
            try:
                self.solver.propagate(changed_arcs)  # Only propagate from changed arcs
            except ValueError:
                return False

        # Check domains non-empty
        for var in self.solver.variables.values():
            if var.domain is not None and len(var.domain) == 0:
                return False
        return True

    def get_possible_positions(self) -> Dict[str, Set[Tuple[int, int]]]:
        # Ensure unconstrained variables have full domain (only those with None)
        for name in self.solver.variables:
            self._ensure_domain_initialized(name)
        return {name: (set(var.domain) if var.domain is not None else set()) for name, var in self.solver.variables.items()}

    def get_num_possible_positions(self) -> Dict[str, int]:
        """Return counts of possible positions for each variable without mutating domains.

        If a variable's domain has not been initialized (None), return the
        full grid cell count as its count. If initialized but empty, return 0.
        """
        full = self.grid_size ** 2
        out: Dict[str, int] = {}
        for name, var in self.solver.variables.items():
            if var.domain is None:
                out[name] = full
            else:
                out[name] = len(var.domain)
        return out

    def get_possible_relations(self, max_samples_per_var: int = 50,
                        perspective: Tuple[int, int] = (0, 1), bin_system=CardinalBinsAllo(), distance_bin_system=StandardDistanceBins(),
                        path_consistent: bool = True) -> Dict[Tuple[str, str], Set[str]]:
        # Possible relationship sets per unordered pair (discrete only)
        names = sorted(self.solver.variables.keys())
        rel_sets: Dict[Tuple[str, str], Set[str]] = {}

        # Ensure all domains initialized if we need path-consistency checks
        if path_consistent:
            for n in names:
                self._ensure_domain_initialized(n)

        def _sample(domain: Set[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
            if len(domain) <= k:
                return list(domain)
            return random.sample(list(domain), k)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                da = _sample(self.solver.variables[a].domain or set(), max_samples_per_var)
                db = _sample(self.solver.variables[b].domain or set(), max_samples_per_var)
                s: Set[str] = set()
                for pa in da:
                    for pb in db:
                        if path_consistent and not self.solver.is_pair_value_path_consistent(a, pa, b, pb):
                            continue
                        rel = PairwiseRelationshipDiscrete.relationship(pa, pb, perspective, bin_system, distance_bin_system)
                        s.add(rel.to_string())
                rel_sets[(a, b)] = s

        return rel_sets

    def _ensure_domain_initialized(self, name: str):
        if self.solver.variables[name].domain is None:
            g = int(self.grid_size)
            self.solver.variables[name].domain = {(x, y) for x in range(-g, g + 1) for y in range(-g, g + 1)}

    def copy(self) -> 'SpatialSolver':
        new_spatial = object.__new__(SpatialSolver)
        new_spatial.grid_size = self.grid_size
        new_spatial.solver = self.solver.copy()
        return new_spatial

    # ---- Path-consistency API ----
    def is_pair_value_path_consistent(self, obj1_name: str, pos1: Tuple[int, int],
                                      obj2_name: str, pos2: Tuple[int, int]) -> bool:
        """PC check for (obj1=pos1, obj2=pos2) against all other variables."""
        # Ensure domains exist for referenced vars (others assumed initialized already)
        self._ensure_domain_initialized(obj1_name)
        self._ensure_domain_initialized(obj2_name)
        return self.solver.is_pair_value_path_consistent(obj1_name, tuple(pos1), obj2_name, tuple(pos2))

    # ---- Metrics ----
    def compute_metrics(self, max_samples_per_var: int = 50,
                        perspective: Tuple[int, int] = (0, 1), bin_system=CardinalBinsAllo(), distance_bin_system=StandardDistanceBins(),
                        path_consistent: bool = True) -> tuple[
        Dict[str, int], int, Dict[Tuple[str, str], Set[str]], int
    ]:
        """
        Compute discrete relationship sets using the provided bin systems.
        Returns: (domain_sizes, total_positions, pair_rel_sets, total_relationships)
        """
        bin_system = bin_system or CardinalBinsAllo()
        distance_bin_system = distance_bin_system or StandardDistanceBins()
        
        # Domain sizes
        domain_sizes: Dict[str, int] = {}
        for name in self.solver.variables:
            self._ensure_domain_initialized(name)
            domain_sizes[name] = len(self.solver.variables[name].domain)
        total_positions = sum(domain_sizes.values())

        # Possible relationship sets per unordered pair (discrete only)
        rel_sets = self.get_possible_relations(max_samples_per_var, perspective, bin_system, distance_bin_system, path_consistent)

        total_relationships = sum(len(v) for v in rel_sets.values())
        return domain_sizes, total_positions, rel_sets, total_relationships
