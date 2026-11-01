# SD-FSNN core source archive
Core source extracted from the SD-FSNN research project.

Gaussian-decay features, GPE coefficient-ODE solvers, Domain and activation helpers, BCL solver definitions.
No baseline implementations, datasets, sampled points, saved coefficients, reference solutions, experimental outputs, figures or pretrained weights. Numeric literals in source are algorithm parameters.

Nine original files are byte-for-byte copies. Three package initializers have narrowed exports to remove unrelated PDE dependencies. bcl_gpe_core.py extracts seven original definitions from examples/GPE_sdre_bcl.py, preserving their bodies; CLI, reference-solver and plotting workflows are omitted. This extraction is source material rather than a standalone experiment driver; optional branches may need original caller configuration/helpers. Third-party swimnetworks is an external dependency and is not bundled.

All Python files passed syntax checks. No numerical experiments were run. This archive does not certify reproduction of paper results or code-to-manuscript equivalence. Add src to PYTHONPATH for use.
