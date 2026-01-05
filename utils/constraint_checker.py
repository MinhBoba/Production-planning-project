"""
Constraint violation checker for Pyomo models.
Useful for debugging infeasible or suboptimal solutions.
"""

import math
import pyomo.environ as pyo
from typing import List, Tuple, Any

try:
    from pyomo.core.base.piecewise import PiecewiseData
except ImportError:
    PiecewiseData = None


def find_violations(
    model: pyo.ConcreteModel,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_lines: int = 50,
    skip_piecewise: bool = True,
) -> List[Tuple]:
    """
    Find constraint and variable bound violations in a Pyomo model.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved Pyomo model to check
    atol : float
        Absolute tolerance for violations
    rtol : float
        Relative tolerance (as fraction of RHS)
    max_lines : int
        Maximum violations to print
    skip_piecewise : bool
        If True, skip helper constraints from Piecewise transformation

    Returns
    -------
    List[Tuple]
        List of violations, each tuple contains:
        (type, name, index, lhs, relation, rhs, gap)

    Notes
    -----
    A violation occurs when:
        |lhs - rhs| > atol + rtol * |rhs|

    Examples
    --------
    >>> violations = find_violations(model, atol=1e-5)
    >>> if violations:
    ...     print(f"Found {len(violations)} violations")
    """

    def inside_piecewise_block(comp):
        """Check if component is inside piecewise transformation."""
        if not skip_piecewise or PiecewiseData is None:
            return False
        blk = comp.parent_block()
        while blk is not None:
            if isinstance(blk, PiecewiseData):
                return True
            blk = blk.parent_block()
        return False

    violations = []
    val = pyo.value

    # Check constraints
    for c in model.component_data_objects(pyo.Constraint, active=True):
        if inside_piecewise_block(c):
            continue

        try:
            lhs = val(c.body) if c.body is not None else 0.0
        except ValueError:
            # Uninitialized expression - skip
            continue

        lo = val(c.lower) if c.has_lb() else -math.inf
        up = val(c.upper) if c.has_ub() else math.inf

        # Check equality constraints
        if c.equality:
            rhs = lo  # For equality, lo == up
            resid = abs(lhs - rhs)
            if resid > atol + rtol * abs(rhs):
                violations.append(
                    (
                        "C",
                        c.parent_component().name,
                        c.index(),
                        lhs,
                        "=",
                        rhs,
                        resid,
                    )
                )
            continue

        # Check lower bound
        if lhs < lo - (atol + rtol * abs(lo)):
            violations.append(
                (
                    "C",
                    c.parent_component().name,
                    c.index(),
                    lhs,
                    ">=",
                    lo,
                    lo - lhs,
                )
            )

        # Check upper bound
        if lhs > up + (atol + rtol * abs(up)):
            violations.append(
                (
                    "C",
                    c.parent_component().name,
                    c.index(),
                    lhs,
                    "<=",
                    up,
                    lhs - up,
                )
            )

    # Check variable bounds
    for v in model.component_data_objects(pyo.Var, active=True):
        if inside_piecewise_block(v):
            continue

        try:
            x = val(v)
        except ValueError:
            # Uninitialized variable - skip
            continue

        lb, ub = v.lb, v.ub
        name = v.parent_component().name
        idx = v.index()

        # Check lower bound
        if lb is not None and x < lb - (atol + rtol * abs(lb)):
            violations.append(("LB", name, idx, x, ">=", lb, lb - x))

        # Check upper bound
        if ub is not None and x > ub + (atol + rtol * abs(ub)):
            violations.append(("UB", name, idx, x, "<=", ub, x - ub))

    # Sort by gap size (largest first)
    violations.sort(key=lambda rec: -rec[-1])

    # Print results
    if not violations:
        print("No violations above tolerance.")
    else:
        print(
            f"{len(violations)} violations "
            f"(showing up to {max_lines}):"
        )
        for k, n, idx, lhs, s, rhs, g in violations[:max_lines]:
            idx_str = "" if idx == () else str(idx)
            print(
                f" {k:<2} {n}{idx_str:<20} : "
                f"{lhs: .6g}  {s}  {rhs: .6g}   (gap = {g: .3g})"
            )

    return violations
