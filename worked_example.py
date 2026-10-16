"""
worked_example.py -- one vertex through every space (build plan phase 6).

Traces the plan's reference vertex (2,3,4) through the whole chain and
prints every intermediate value, so the pinned conventions can be checked
by hand:

    object -> world    M = rot_y(90) @ translate(1, 0, -1)
                       ("translate then rotate": the right-most matrix
                        acts first)
    world  -> eye      V = look_at(eye, target, up)
    eye    -> clip     P = perspective(fovy_deg, aspect, near, far)
    clip   -> NDC      divide by w
    NDC    -> screen   viewport(800, 600), y flipped, depth (z + 1) / 2

Reference run (pinned in PLAN.md phase 6, and re-checked here):

    world  = (3, 3, -3, 1)
    eye    = (3, 3, -11, 1)
    clip   = (5.196, 5.196, 9.408, 11)
    ndc    = (0.472, 0.472, 0.855)
    screen = (589, 158), depth 0.928

The script also re-runs the chain through transforms.project() and
compares it with the manual step-by-step matrices -- the renderers use
project(), so this pins both code paths to the same numbers.

Run:  python worked_example.py
Exit code 0 means every value matches the reference table.
"""

import sys

import numpy as np

from transforms import look_at, perspective, project, rot_y, translate, viewport

# The reference scenario, verbatim from the plan.
POINT = (2.0, 3.0, 4.0)
EYE = (0.0, 0.0, 8.0)
TARGET = (0.0, 0.0, 0.0)
UP = (0.0, 1.0, 0.0)
FOVY_DEG = 60.0
NEAR = 1.0
FAR = 50.0
WIDTH, HEIGHT = 800, 600

# The pinned reference values (rounded as the plan prints them).
REFERENCE_WORLD = (3.0, 3.0, -3.0, 1.0)
REFERENCE_EYE = (3.0, 3.0, -11.0, 1.0)
REFERENCE_CLIP = (5.196, 5.196, 9.408, 11.0)
REFERENCE_NDC = (0.472, 0.472, 0.855)
REFERENCE_PIXEL = (589.0, 158.0)
REFERENCE_DEPTH01 = 0.928

# Bookkeeping for the exit code (and for run(verbose=False) quiet mode).
_TOTAL = 0
_FAILURES = []
_QUIET = False


def _fmt(v, decimals=4):
    """Format a vector as (a, b, c, ...) with a fixed number of decimals."""
    return "(" + ", ".join(f"{float(x):.{decimals}f}" for x in np.asarray(v).ravel()) + ")"


def _check(name, actual, expected, atol=1e-9):
    """Print one comparison against the reference table; count failures."""
    global _TOTAL
    _TOTAL += 1
    ok = np.allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=atol,
    )
    if not _QUIET:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         actual   = {_fmt(actual)}")
            print(f"         expected = {_fmt(expected)}")
    if not ok:
        _FAILURES.append(name)
    return ok


def run(verbose=True):
    """Do the reference trace, optionally print it, count mismatches.

    Returns the number of values that do NOT match the plan's reference
    table; 0 means the phase-6 check passed. tests.py calls this with
    verbose=False so the suite stays quiet.
    """
    global _QUIET
    _QUIET = not verbose

    # The three matrices of the chain (object -> world -> eye -> clip).
    model = rot_y(90) @ translate(1, 0, -1)
    view = look_at(EYE, TARGET, UP)
    proj = perspective(FOVY_DEG, 1.0, NEAR, FAR)

    # Manual step-by-step chain, one matrix at a time.
    p = np.array([*POINT, 1.0])
    world = model @ p
    eye = view @ world
    clip = proj @ eye
    ndc = clip[:3] / clip[3]
    screen = viewport(ndc, WIDTH, HEIGHT)[0]  # (3,): x_pix, y_pix, depth01

    # The renderers' code path: transforms.project() composes P @ V @ M
    # once and multiplies the whole batch -- same numbers expected.
    clip_batch, ndc_batch = project(proj, view, model, np.array([POINT]))
    chain_ok = np.allclose(clip_batch[0], clip, rtol=0.0, atol=1e-9) and np.allclose(
        ndc_batch[0], ndc, rtol=0.0, atol=1e-9
    )

    if verbose:
        print("worked_example: one vertex through every space (PLAN.md phase 6)")
        print()
        print("inputs")
        print(f"  point          = {_fmt(POINT, 0)}")
        print("  model          = rot_y(90) @ translate(1, 0, -1)")
        print(f"  camera         = eye {_fmt(EYE, 0)}, target {_fmt(TARGET, 0)}, up {_fmt(UP, 0)}")
        print(
            f"  projection     = perspective({FOVY_DEG:g} deg, aspect 1, "
            f"near {NEAR:g}, far {FAR:g})"
        )
        print(f"  framebuffer    = {WIDTH} x {HEIGHT}")
        print()
        print("matrices")
        print("  M = rot_y(90) @ translate(1, 0, -1) =")
        print(np.array2string(model, precision=3, suppress_small=True))
        print("  V = look_at(eye, target, up) =")
        print(np.array2string(view, precision=3, suppress_small=True))
        print("  P = perspective(60, 1, 1, 50) =")
        print(np.array2string(proj, precision=3, suppress_small=True))
        print()
        print("steps")
        print(f"  object  p              = {_fmt(p)}   [given]")
        print(f"  world   M @ p          = {_fmt(world)}   translate then rotate")
        print(
            f"  eye     V @ world      = {_fmt(eye)}   camera at (0,0,8) "
            "looking down -z"
        )
        print(f"  clip    P @ eye        = {_fmt(clip)}   w = -z_eye = 11")
        print(f"  ndc     clip.xyz / w   = {_fmt(ndc)}   the perspective divide")
        print(
            f"  screen  viewport(...)  = {_fmt(screen)}   x scales to 800, "
            "y flips (screen y grows down)"
        )
        print(
            f"  pixel   rounded        = "
            f"({float(np.round(screen[0])):.0f}, {float(np.round(screen[1])):.0f}), "
            f"depth {np.round(screen[2], 3):.3f}"
        )
        print()
        print("checks vs the pinned reference table")

    # Rounded exactly like the plan prints them; the tolerance compares
    # the rounded values (e.g. 5.196152... -> 5.196) against the table.
    _check("world  = (3, 3, -3, 1)", np.round(world, 3), REFERENCE_WORLD)
    _check("eye    = (3, 3, -11, 1)", np.round(eye, 3), REFERENCE_EYE)
    _check("clip   = (5.196, 5.196, 9.408, 11)", np.round(clip, 3), REFERENCE_CLIP)
    _check("ndc    = (0.472, 0.472, 0.855)", np.round(ndc, 3), REFERENCE_NDC)
    _check(
        "screen = (589, 158)",
        np.round(screen[:2]),
        REFERENCE_PIXEL,
    )
    _check("depth  = 0.928", np.round(screen[2], 3), REFERENCE_DEPTH01)
    _check(
        "project() agrees with the manual chain",
        np.array([int(chain_ok)]),
        np.array([1]),
    )

    if verbose:
        print()
        if _FAILURES:
            print(f"result: {len(_FAILURES)} of {_TOTAL} checks FAILED")
        else:
            print(f"result: all {_TOTAL} reference values match -- phase 6 check passed")
    return len(_FAILURES)


def main():
    """CLI: run the worked example; exit 0 when the reference matches."""
    return 1 if run(verbose=True) else 0


if __name__ == "__main__":
    sys.exit(main())
