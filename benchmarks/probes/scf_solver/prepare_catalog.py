"""Fetch and validate public SCF fixtures, without importing chemistry packages.

Usage: python prepare_catalog.py [--only ID ...] [--validate-only]
Cached primary-source responses are retained; this never executes SCF or builds
an artificial Cartesian product of systems, basis sets, methods or functionals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote

if __package__:
    from .geometry_construction import archived_water, metal_water, validate_coordination
else:
    from geometry_construction import archived_water, metal_water, validate_coordination

ROOT = Path(__file__).resolve().parent
Z = {symbol: number for number, symbol in enumerate(
    ("", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar")) if symbol}
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def formula_counts(formula):
    return Counter({element: int(number or 1) for element, number in
                    re.findall(r"([A-Z][a-z]?)(\d*)", formula)})


def fetch(url, path):
    if not path.exists():
        result = subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "2",
            "--retry-delay", "1", "--max-time", "35", url], capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode().strip())
        path.write_bytes(result.stdout)
        time.sleep(.25)  # Keep the sequential PUG-REST request rate bounded.
    return path.read_bytes()


def parse_sdf(data, *, require_marker=True):
    lines = data.decode().splitlines()
    if "V2000" not in lines[3] or (require_marker and "3D" not in lines[1]):
        raise ValueError("Expected an explicitly three-dimensional V2000 record")
    count, bonds = int(lines[3][:3]), int(lines[3][3:6])
    atoms = [(line[31:34].strip(), *(float(line[j:j+10]) for j in (0, 10, 20)))
             for line in lines[4:4+count]]
    if len(atoms) != count or any(a[0] not in Z or not all(math.isfinite(x) for x in a[1:]) for a in atoms):
        raise ValueError("Unsupported atom or incomplete/nonfinite coordinates")
    if any(math.dist(a[1:], b[1:]) < .2 for i, a in enumerate(atoms) for b in atoms[:i]):
        raise ValueError("Coincident atoms in the source geometry")
    charges = {i: {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}.get(int(line[36:39]), 0)
               for i, line in enumerate(lines[4:4+count], 1)}
    radicals = []
    for line in lines[4+count+bonds:]:
        if line.startswith("M  CHG"):
            fields = list(map(int, line[6:].split()))
            charges.update(zip(fields[1::2], fields[2::2]))
        if line.startswith("M  RAD"):
            radicals.append(line)
    return atoms, sum(charges.values()), radicals


def water_cluster(seed, atoms, sdf, source):
    """Add only rigid translations to the two archived neutral water fragments."""
    water, water_source = archived_water(ROOT.parents[2])
    xmax, oxygen = max(a[1] for a in atoms), next(a[1:] for a in water if a[0] == "O")
    fragments = [atoms]
    for dx in (8., 16.):
        shift = (xmax+dx-oxygen[0], -oxygen[1], -oxygen[2])
        fragments.append([(symbol, *(x+d for x, d in zip(coord, shift))) for symbol, *coord in water])
    separation = min(math.dist(a[1:], b[1:]) for i, frag in enumerate(fragments)
        for other in fragments[:i] for a in frag for b in other)
    assert separation >= 5., "Rigid fragments must not overlap"
    combined = sum(fragments, [])
    lines = sdf.decode().splitlines()
    n, bonds = int(lines[3][:3]), int(lines[3][3:6])
    atom_lines = lines[4:4+n] + [f"{x:10.4f}{y:10.4f}{z:10.4f} {s:<3} 0  0  0  0  0  0" for s, x, y, z in combined[n:]]
    bond_lines = lines[4+n:4+n+bonds]
    for offset in (n, n+3):
        bond_lines += [f"{offset+1:3d}{offset+j:3d}  1  0  0  0  0" for j in (2, 3)]
    records = lines[4+n+bonds:lines.index("M  END")+1]
    data = ("\n".join([seed["id"], "  GradSCF rigid benchmark3D", "",
        f"{len(combined):3d}{len(bond_lines):3d}  0  0  0  0  0  0  0  0999 V2000", *atom_lines, *bond_lines, *records, "$$$$", ""])).encode()
    path = ROOT/"geometries"/(seed["id"]+".sdf")
    path.write_bytes(data)
    source.update(parent_geometry_file=source["geometry_file"], parent_geometry_sha256=source["geometry_sha256"],
        **water_source,
        construction=dict(water_oxygen_x_offsets_from_drug_max_angstrom=[8., 16.],
            cross_fragment_min_distance_angstrom=separation, optimization="none"),
        geometry_file=str(path.relative_to(ROOT)), geometry_sha256=digest(data),
        geometry_url=None, parent_geometry_url=source["geometry_url"])
    return combined, source


def catalog_row(seed, atoms, source, name):
    if not 5 <= len(atoms) <= 150 or Counter(a[0] for a in atoms) != formula_counts(seed["expected_formula"]):
        raise ValueError("Atom count/formula differs from the declared5--150 atom system")
    electrons = sum(Z[a[0]] for a in atoms)-seed["charge"]
    if (electrons-seed["spin"]) % 2 or electrons < seed["spin"]:
        raise ValueError("Electron count and explicit spin have inconsistent parity")
    decimals = 12 if source["geometry_file"].endswith(".xyz") else 4
    return dict(id=seed["id"], name=name, formula=seed["expected_formula"],
        basis=seed["basis"], charge=seed["charge"], spin=seed["spin"],
        natom=len(atoms), elements=sorted({a[0] for a in atoms}, key=Z.get),
        nelectron=electrons, atom="; ".join(f"{s} {x:.{decimals}f} {y:.{decimals}f} {z:.{decimals}f}" for s, x, y, z in atoms),
        unit="Angstrom", cart=True, geometry_role=seed["geometry_role"], source=source)


def prepare_coordination(seed):
    atoms, construction = metal_water(seed, ROOT.parents[2])
    basis_path = ROOT.parents[2]/seed["basis_file"]
    if not re.search(r"^"+seed["ideal_metal"]+r"\s+[SPDF]", basis_path.read_text(), re.MULTILINE):
        raise ValueError("Assigned basis snapshot lacks the requested metal")
    definition = dict(id=seed["id"], formula=seed["expected_formula"], charge=seed["charge"], spin=0,
        construction=construction, metal_atomic_number=Z[seed["ideal_metal"]],
        basis_source=seed["basis_file"], basis_source_sha256=digest(basis_path.read_bytes()),
        construction_code_sha256=digest(Path(__file__).with_name("geometry_construction.py").read_bytes()))
    xyz = ROOT/"geometries"/(seed["id"]+".xyz")
    xyz.write_text(f"{len(atoms)}\nUnoptimized ideal cation-water computational benchmark; Angstrom\n"+
        "\n".join(f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, x, y, z in atoms)+"\n")
    actual = [(p[0], *map(float, p[1:])) for p in (line.split() for line in xyz.read_text().splitlines()[2:])]
    definition["serialized_geometry_validation"] = validate_coordination(actual, seed, ROOT.parents[2])
    identity = ROOT/"sources"/(seed["id"]+".json")
    identity.write_text(json.dumps(definition, indent=2)+"\n")
    source = dict(provider="Constructed coordination benchmark", component_id=seed["id"],
        title=seed["name"], identity_url=None, identity_file=str(identity.relative_to(ROOT)), identity_sha256=digest(identity.read_bytes()),
        geometry_url=None, geometry_file=str(xyz.relative_to(ROOT)), geometry_sha256=digest(xyz.read_bytes()), **definition)
    return catalog_row(seed, atoms, source, seed["name"])


def prepare(seed):
    if seed.get("ideal_metal"):
        return prepare_coordination(seed)
    name = seed["name"]
    ccd = seed.get("rcsb_id")
    if ccd:
        prop_url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd}"
        lookup = ROOT/"sources"/f"ccd-{ccd}.json"
        raw = fetch(prop_url, lookup)
        component = json.loads(raw)["chem_comp"]
        prop = dict(MolecularFormula=component["formula"].replace(" ", ""),
            Charge=component["pdbx_formal_charge"], Title=component["name"], IUPACName=None, InChIKey=None)
        source_id = dict(provider="RCSB CCD", component_id=ccd)
        sdf_url = f"https://files.rcsb.org/ligands/download/{ccd}_ideal.sdf"
        sdf_path = ROOT/"geometries"/f"ccd-{ccd}-ideal.sdf"
    else:
        prop_url = f"{BASE}/name/{quote(name, safe='')}/property/Title,IUPACName,MolecularFormula,Charge,InChIKey/JSON"
        lookup = ROOT/"sources"/(seed["id"]+".json")
        raw = fetch(prop_url, lookup)
        properties = json.loads(raw)["PropertyTable"]["Properties"]
        if len(properties) != 1:
            raise ValueError("Ambiguous name: pin a verified compound identifier")
        prop = properties[0]
        source_id = dict(provider="PubChem", cid=prop["CID"])
        sdf_url = f"{BASE}/cid/{prop['CID']}/SDF?record_type=3d"
        sdf_path = ROOT/"geometries"/f"pubchem-{prop['CID']}.sdf"
    if formula_counts(prop["MolecularFormula"]) != formula_counts(seed.get("source_formula", seed["expected_formula"])):
        raise ValueError(f"Name resolved to unexpected formula {prop['MolecularFormula']}")
    sdf = fetch(sdf_url, sdf_path)
    atoms, source_charge, radicals = parse_sdf(sdf, require_marker=not ccd)
    if source_charge != prop["Charge"] or Counter(a[0] for a in atoms) != formula_counts(prop["MolecularFormula"]):
        raise ValueError("SDF charge/formula disagrees with the identity lookup")
    source = dict(**source_id, title=prop["Title"], iupac_name=prop["IUPACName"],
        inchikey=prop["InChIKey"], source_charge=prop["Charge"], radical_records=radicals,
        identity_url=prop_url, identity_file=str(lookup.relative_to(ROOT)), identity_sha256=digest(raw),
        geometry_url=sdf_url, geometry_file=str(sdf_path.relative_to(ROOT)), geometry_sha256=digest(sdf))
    if seed.get("cluster_waters"):
        atoms, source = water_cluster(seed, atoms, sdf, source)
        if Counter(a[0] for a in atoms) != formula_counts(seed["expected_formula"]):
            raise ValueError("Constructed cluster formula differs from the declared components")
    if prop["Charge"] != seed["charge"] and "geometry_role" not in seed:
        raise ValueError("A charge change requires an explicit vertical-state label")
    label = (name + " radical " + ("cation" if seed["charge"] > 0 else "anion")
             if seed["charge"] != prop["Charge"] else name)
    seed = dict(seed, geometry_role=seed.get("geometry_role", "RCSB CCD ideal coordinates; no ab initio optimization or experimental geometry claimed" if ccd else
        "PubChem computed 3D conformer; no ab initio optimization or experimental geometry claimed"))
    return catalog_row(seed, atoms, source, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    seeds = json.loads((ROOT/"catalog_seed.json").read_text())["systems"]
    for name in ("sources", "geometries"):
        (ROOT/name).mkdir(exist_ok=True)
    if args.validate_only:
        rows = json.loads((ROOT/"systems.json").read_text())["systems"]
        assert len(rows) == len({r["id"] for r in rows}) == 60
        assert set(Counter(r["basis"] for r in rows).values()) == {12}
        for row in rows:
            path = ROOT/row["source"]["geometry_file"]
            if path.suffix == ".xyz":
                lines = path.read_text().splitlines()
                atoms = [(p[0], *map(float, p[1:])) for p in (line.split() for line in lines[2:])]
                assert len(atoms) == int(lines[0])
                seed = next(s for s in seeds if s["id"] == row["id"])
                expected, _ = metal_water(seed, ROOT.parents[2])
                assert max(math.dist(a[1:], b[1:]) for a, b in zip(atoms, expected)) < 1e-11
                validate_coordination(atoms, seed, ROOT.parents[2])
            else:
                atoms, _, _ = parse_sdf(path.read_bytes(), require_marker=row["source"]["provider"] == "PubChem")
            assert len(atoms) == row["natom"]
            assert digest((ROOT/row["source"]["geometry_file"]).read_bytes()) == row["source"]["geometry_sha256"]
            assert digest((ROOT/row["source"]["identity_file"]).read_bytes()) == row["source"]["identity_sha256"]
            assert Counter(a[0] for a in atoms) == formula_counts(row["formula"])
            assert sum(Z[a[0]] for a in atoms)-row["charge"] == row["nelectron"]
            assert (row["nelectron"]-row["spin"]) % 2 == 0
            decimals = 12 if path.suffix == ".xyz" else 4
            assert "; ".join(f"{s} {x:.{decimals}f} {y:.{decimals}f} {z:.{decimals}f}" for s, x, y, z in atoms) == row["atom"]
        assert min(r["natom"] for r in rows) == 5 and max(r["natom"] for r in rows) == 150
        print("Validated60 systems,12 per primary basis; no numerical calculations.")
        return
    existing = ROOT/"systems.json"
    rows = {r["id"]: r for r in json.loads(existing.read_text())["systems"]} if existing.exists() else {}
    failures = []
    for seed in seeds:
        if args.only and seed["id"] not in args.only:
            continue
        try:
            rows[seed["id"]] = prepare(seed)
            print(seed["id"], rows[seed["id"]]["natom"], "atoms", flush=True)
        except (RuntimeError, ValueError, KeyError, IndexError) as exc:
            rows.pop(seed["id"], None)  # Never retain stale data after a failed refresh.
            failures.append(dict(id=seed["id"], error=str(exc)))
            print(seed["id"], "UNAVAILABLE", str(exc), flush=True)
    result = dict(schema_version=1, prepared_at=datetime.now(timezone.utc).isoformat(),
        scope="60 distinct physical species,12 per primary basis; no method/basis Cartesian product",
        protocol=dict(dtype="float64", unit="Angstrom", cart=True, dft_grid_level=3,
            method_families=["RHF", "UHF", "GHF", "RKS", "UKS", "GKS"],
            functionals=[dict(family=family, gradscf=xc, pyscf=reference) for family, xc, reference in
                (("LDA", "svwn", "lda,vwn"), ("GGA", "pbe", "pbe"),
                 ("global_hybrid", "pbe0", "pbe0"), ("RSH", "cam-b3lyp", "cam-b3lyp"))],
            cross_basis_system_ids=["methane", "benzene", "phenanthrene"],
            cross_basis_note="Optional representative controls, recorded separately from the60 primary cases; never expand all60 across all5 basis sets.",
            boundaries=["RHF/RKS are not applicable to the declared doublet cases.",
                "GKS RSH is currently unsupported; do not substitute UKS.",
                "GHF/GKS currently require full/packed AO ERIs; DF/direct are unsupported.",
                "Run-time AO dimensions, memory forecasts and resource/convergence gates belong to the runner; this catalog is not a result."]),
        systems=[rows[s["id"]] for s in seeds if s["id"] in rows], unavailable=failures)
    existing.write_text(json.dumps(result, indent=2)+"\n")
    print("Prepared",len(result["systems"]),"systems; unavailable",len(failures))


if __name__ == "__main__":
    main()
