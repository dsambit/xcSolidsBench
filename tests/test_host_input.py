"""pw.x input generation and pseudopotential checks (laptop; needs XCSB_PSEUDO_DIR with the SG15 files)."""
import os

import pytest

from xcsb.bench.structures import bcc, rocksalt_afm2
from xcsb.host.qe import Case, pw_input
from xcsb.pseudo import PseudoError, check_element

PDIR = os.environ.get("XCSB_PSEUDO_DIR")
needs_pseudos = pytest.mark.skipif(not PDIR or not os.path.isdir(PDIR), reason="XCSB_PSEUDO_DIR not set")


@needs_pseudos
def test_fe_fm_input():
    c = Case(bcc("Fe", 5.42), ecutwfc=90.0, kmesh=(12, 12, 12), nspin=2, starting_magnetization={"Fe": 0.6})
    txt = pw_input(c, "TPSS", PDIR, "/tmp/out", "x")
    assert "nspin = 2" in txt and "starting_magnetization(1) = 0.600000" in txt and "ecutrho = 360.000000" in txt
    assert "input_dft = 'TPSS'" in txt and "smearing = 'gaussian'" in txt and "conv_thr = 1.000e-08" in txt and "tstress = .false." in txt


@needs_pseudos
def test_afm2_species():
    s = rocksalt_afm2("Ni", "O", 7.88)
    c = Case(s, ecutwfc=100.0, nspin=2, starting_magnetization={"Ni1": 0.5, "Ni2": -0.5, "O": 0.0})
    txt = pw_input(c, "TPSS", PDIR, "/tmp/out", "x")
    assert "ntyp = 3" in txt and "starting_magnetization(2) = -0.500000" in txt
    assert "  Ni1 1.0 Ni_ONCV_PBE-1.2.upf" in txt and "  Ni2 1.0 Ni_ONCV_PBE-1.2.upf" in txt


def test_nspin2_needs_start():
    with pytest.raises(ValueError):
        Case(bcc("Fe", 5.42), ecutwfc=90.0, nspin=2)


def test_noncollinear_refused():
    with pytest.raises(ValueError):
        Case(bcc("Fe", 5.42), ecutwfc=90.0, nspin=4)


@needs_pseudos
def test_sha_mismatch_raises(tmp_path):
    src = os.path.join(PDIR, "Fe_ONCV_PBE-1.2.upf")
    bad = tmp_path / "Fe_ONCV_PBE-1.2.upf"
    bad.write_text(open(src).read() + "\n")
    with pytest.raises(PseudoError):
        check_element("Fe", str(tmp_path))


@needs_pseudos
def test_restart_input():
    """A production warm start (startingwfc and startingpot 'file') of an FSM point."""
    c = Case(bcc("Fe", 5.42), ecutwfc=80.0, kmesh=(8, 8, 8), degauss=0.02, nspin=2, starting_magnetization={"Fe": 0.5},
             tot_magnetization=2.8)
    txt = pw_input(c, "TPSS", PDIR, "/tmp/out", "xcsb", restart=True)
    assert "startingwfc = 'file'" in txt and "startingpot = 'file'" in txt and "tot_magnetization = 2.800000" in txt
