"""Crystal cells (DESIGN.md Sec. 2, 6.6). Lattice rows in bohr, fractional positions, one species label per atom.

Species labels are distinct names for the same element when sites must carry different starting magnetizations
(antiferromagnets: e.g. "Ni1", "Ni2"); every label maps to its element (and so to its pseudopotential).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


@dataclass(frozen=True)
class Structure:
    name: str
    lattice: np.ndarray            # (3, 3) rows a_i, bohr
    labels: tuple                  # per atom: species label
    frac: np.ndarray               # (nat, 3) fractional coordinates
    elements: dict                 # label -> element symbol

    @property
    def nat(self):
        return len(self.labels)

    @property
    def species(self):
        """Distinct labels in first-appearance order."""
        out = []
        for lab in self.labels:
            if lab not in out:
                out.append(lab)
        return out

    @property
    def volume(self):
        return abs(float(np.linalg.det(self.lattice)))

    def cartesian(self):
        return self.frac @ self.lattice


def _s(name, lattice, labels, frac, elements=None):
    labels = tuple(labels)
    elements = elements or {lab: lab for lab in labels}
    return Structure(name, np.asarray(lattice, dtype=float), labels, np.asarray(frac, dtype=float), dict(elements))


def bcc(el, a):
    return _s(f"bcc-{el}", 0.5 * a * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]), [el], [[0, 0, 0]])


def fcc(el, a):
    return _s(f"fcc-{el}", 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), [el], [[0, 0, 0]])


def diamond(el, a):
    return _s(f"diamond-{el}", 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), [el, el], [[0, 0, 0], [0.25] * 3])


def zincblende(a_el, b_el, a):
    return _s(f"zb-{a_el}{b_el}", 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), [a_el, b_el],
              [[0, 0, 0], [0.25] * 3])


def rocksalt(a_el, b_el, a):
    return _s(f"rs-{a_el}{b_el}", 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), [a_el, b_el],
              [[0, 0, 0], [0.5] * 3])


def cesium_chloride(a_el, b_el, a):
    return _s(f"cscl-{a_el}{b_el}", a * np.eye(3), [a_el, b_el], [[0, 0, 0], [0.5] * 3])


def hcp(el, a, c):
    lat = np.array([[a, 0, 0], [-0.5 * a, 0.5 * np.sqrt(3) * a, 0], [0, 0, c]])
    return _s(f"hcp-{el}", lat, [el, el], [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]])


def rocksalt_afm2(tm, o, a):
    """Rocksalt TM-O with AFM-II order: ferromagnetic (111) planes stacked antiferromagnetically along [111].
    Rhombohedral cell a1 = a(1, 1/2, 1/2), a2 = a(1/2, 1, 1/2), a3 = a(1/2, 1/2, 1) (volume a^3/2, two formula units);
    TM1 at 0, TM2 at (a1+a2+a3)/2, O at (a1+a2+a3)/4 and 3(a1+a2+a3)/4. Labels tm+'1', tm+'2' (opposite moments)."""
    lat = a * np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    labels = [tm + "1", tm + "2", o, o]
    frac = [[0, 0, 0], [0.5] * 3, [0.25] * 3, [0.75] * 3]
    return _s(f"afm2-{tm}{o}", lat, labels, frac, {tm + "1": tm, tm + "2": tm, o: o})


def neighbours(s: Structure, i: int, rmax: float):
    """(distance, label) of every atom within rmax of atom i (periodic images), sorted by distance."""
    cart = s.cartesian()
    out = []
    rng = range(-3, 4)
    for n1 in rng:
        for n2 in rng:
            for n3 in rng:
                shift = n1 * s.lattice[0] + n2 * s.lattice[1] + n3 * s.lattice[2]
                for j in range(s.nat):
                    d = float(np.linalg.norm(cart[j] + shift - cart[i]))
                    if 1e-8 < d <= rmax:
                        out.append((d, s.labels[j]))
    return sorted(out)
