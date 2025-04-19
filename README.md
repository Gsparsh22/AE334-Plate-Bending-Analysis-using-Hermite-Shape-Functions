# AE334 Plate Bending Analysis using Hermite Shape Functions

This project provides a Python implementation for the bending analysis of rectangular plates using Hermite shape functions. The code computes deflections, stresses, and strain energy for plates under uniform loading, supporting various boundary conditions and polynomial orders.

## Features

- **Hermite Shape Functions**: Implements cubic and higher-order Hermite shape functions for plate bending.
- **Boundary Conditions**: Supports simply-supported (SSSS) and clamped-free (CFFF) edge conditions.
- **Finite Element Assembly**: Assembles global stiffness and force matrices using Gauss quadrature.
- **Postprocessing**: Plots deflection surfaces, contours, von Mises stress, and through-thickness stress profiles.
- **Yield Analysis**: Computes maximum von Mises stress and estimates yield load.
- **Visualization**: Saves all plots in a `Plots` directory.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

Install dependencies with:
```bash
pip install numpy pandas matplotlib
```

## Usage

Run the main script:
```bash
python <script_name>.py
```
Replace `<script_name>.py` with the actual filename.

All output plots will be saved in the `Plots` folder.

## Main Functions

- `assemble_plate_system`: Builds the global stiffness matrix and force vector.
- `compute_plate_deflection`: Evaluates plate deflection at grid points.
- `evaluate_derivatives`: Computes second derivatives for stress calculation.
- Plotting functions: Generate surface, contour, von Mises, and through-thickness stress plots.

## Customization

- **Plate Properties**: Modify `E`, `neu`, `h`, `a`, `b`, and `qo` in the `main()` function.
- **Polynomial Order**: Change `p_values` for different Hermite polynomial degrees.
- **Boundary Conditions**: Edit `boundary_conditions` for different edge supports.

## Output

- Deflection surface and contour plots
- Von Mises stress contour
- Through-thickness stress profiles at the plate center
- Console output for maximum von Mises stress, yield load, and strain energy

## References

- Timoshenko, S.P., Woinowsky-Krieger, S. "Theory of Plates and Shells"
- Cook, R.D., Malkus, D.S., Plesha, M.E. "Concepts and Applications of Finite Element Analysis"

---

*For academic use in AE334 or similar courses. Contributions and suggestions are welcome!*