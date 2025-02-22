
# Saturated Absorption Spectroscopy Analysis Code

Code for analysis of my saturated absorption spectroscopy experiment for PHYS330: Advanced Laboratory. 


### NumericalDiagonalization.py
This code numerically diagonalizes the hyperfine and Zeeman Hamiltonian to calculate Zeeman energy spacings as a function of magnetic field. 

### HyperfinePeakGUI.py
This provides a GUI for adjusting SAS peak finding settings and manually verifying the correct peaks are being tracked. The output file can be analyzed using code in DataAnalysis.ipynb to generate values for the hyperfine spacing, including uncertainty.

### ZeemanPeakGUI.py
This GUI is used for hand identifying Zeeman transition peaks. A quadratic curve is then fit in the neighborhood around each peak in order to refine its location. The output file can be read by Data_vs_Model.py to compare numerical results to experiment.

### Data_vs_Model.py
This script plots experimental data vs numerical results.

### DataAnalysis.ipynb
This file contains smaller scripts used in analysis. It also contains code for generating all other figures from the paper.
