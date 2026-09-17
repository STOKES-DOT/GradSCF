# qavg-vSZPs data provenance

Basis/ECP data by Benedikt Bädorf, Marcel Müller, Thomas Froitzheim and Stefan
Grimme, distributed under Creative Commons Attribution 4.0 International.
The original license is in LICENSE. Source: https://github.com/grimme-lab/q-vSZPs
at commit 0eb31c20d7d5044fc44221f967279b3d040881f1.

Reference: q-vSZPs: A polarized, adaptive minimal Gaussian basis set for the
elements Z = 1–86 designed for efficient mean-field electronic structure
calculations. DOI: 10.1063/5.0345149.

The ORCA-format averaged basis and scalar ECP file are copied byte-for-byte;
provenance.json records upstream Git blob and SHA-256 hashes. GradSCF parses
these data without a runtime external chemistry library. The main-group
subset through Xe is exposed by the current neural adapter.

This is a neural variant of the q-vSZPs primitive/ECP design. Average coefficients
initialize a trainable output bias only. Forward evaluation directly predicts
coefficients; it contains no reference-plus-delta or empirical charge/CN formula.

The separately named adaptive q-vszps_basis file at this upstream commit has
parent-like primitive counts (e.g. H 8s3p), inconsistent with the q-vSZPs paper
and qavg-vSZPs file. It is deliberately not imported, and this implementation
does not claim to reproduce the original adaptive charge model.
