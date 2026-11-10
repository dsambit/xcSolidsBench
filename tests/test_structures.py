"""Cells: volumes and the AFM-II neighbour shells (laptop)."""
import numpy as np

from xcsb.bench.structures import bcc, diamond, fcc, hcp, neighbours, rocksalt, rocksalt_afm2


def test_volumes():
    a = 5.0
    assert np.isclose(bcc("Fe", a).volume, a ** 3 / 2)
    assert np.isclose(fcc("Ni", a).volume, a ** 3 / 4)
    assert np.isclose(diamond("Si", a).volume, a ** 3 / 4)
    assert np.isclose(rocksalt_afm2("Ni", "O", a).volume, a ** 3 / 2)
    assert np.isclose(hcp("Co", a, 1.62 * a).volume, np.sqrt(3) / 2 * a * a * 1.62 * a)


def test_afm2_shells():
    """Each TM: 6 O at a/2; 12 TM at a/sqrt(2) (6 parallel in its (111) plane, 6 antiparallel); 6 TM at a, all
    antiparallel (the 180-degree superexchange partners of AFM-II)."""
    a = 4.17
    s = rocksalt_afm2("Ni", "O", a)
    for i in (0, 1):
        own = s.labels[i]
        nb = neighbours(s, i, 1.01 * a)
        d = np.array([x[0] for x in nb])
        lab = [x[1] for x in nb]
        sh1 = [l for x, l in zip(d, lab) if abs(x - a / 2) < 1e-8]
        sh2 = [l for x, l in zip(d, lab) if abs(x - a / np.sqrt(2)) < 1e-8]
        sh3 = [l for x, l in zip(d, lab) if abs(x - a) < 1e-8]
        assert sh1 == ["O"] * 6
        assert len(sh2) == 12 and sh2.count(own) == 6
        assert len(sh3) == 6 and own not in sh3


def test_rocksalt_shell():
    a = 4.0
    nb = neighbours(rocksalt("Mg", "O", a), 0, 0.51 * a)
    assert [l for _, l in nb] == ["O"] * 6
