import numpy as np
import matplotlib.pyplot as plt

### Numerically diagonalize the RB87 Hyperfine + Zeeman Hamiltonian.

hbar = 1.054571817e-34  #Reduced Planck's constant (Js)
Amp = 1e-6
mu_B = 9.27400899e-24 * 1e-4 # Bohr Magneton (J/G)

A_S1 = 2*np.pi*hbar*(3.417341305452e9) # Magnetic dipole constant (F=1)
A_P2 = 2*np.pi*hbar*(84.718520e6) # Magnetic Dipole Constant (F = 2) --
B_P2 = 2*np.pi*hbar*(12.496537e6) # Electric Quadrapole constant (F=2) --
g_I = -0.000995141410 # Nuclear G factor
g_J_S1 = 2.0023311320 # Lande G factor S1/2
g_J_P2 = 1.336213 # Lande G factor P3/2

# Angular momentu of S1/2 and P3/2
J_S1 = 1/2
J_P2 = 3/2
I = 3/2

def spin_mat(s): # Generate spin matricies through raising and lowering operators
    m = np.arange(-s ,s, 1)
    sp = np.sqrt(s*(s+1) - m*(m+1))

    S_p = np.diag(sp, 1)
    S_m = S_p.T

    S_x = (S_p + S_m)/2
    S_y = (S_p - S_m)/(2*1j)
    S_z = np.diag(np.arange(-s, s+1, 1)[::-1])

    return S_x, S_y, S_z


SJx, SJy, SJz = spin_mat(J_S1)
SIx, SIy, SIz = spin_mat(I)

# Spin matrices are not the same size, and don't properly act on the entire space.
# Using the Kroneker product with the identity of size 2*I + 1 or 2*J + 1 ensures both matrices are the same size,
# while maintaining their actions on their respective subspaces.

Jx = np.kron(np.identity(int(2*I+1)), SJx)
Jy = np.kron(np.identity(int(2*I+1)), SJy)
Jz = np.kron(np.identity(int(2*I+1)), SJz)

Ix = np.kron(SIx, np.identity(int(2*J_S1 + 1)))
Iy = np.kron(SIy, np.identity(int(2*J_S1 + 1)))
Iz = np.kron(SIz, np.identity(int(2*J_S1 + 1)))

# Define the magnetic field to evaluate over
points = 5000
B = np.linspace(0, 500, points)

# S1/2 Zeeman splitting
energy = []

# Hyperfine Hamiltonian for S1/2
ground_HF_H = A_S1*(np.dot(Ix, Jx) + np.dot(Iy, Jy) + np.dot(Iz,Jz))
for b in B:

    # Now include Zeeman terms and rescale
    ground_Z_H = (Amp/(2*np.pi*hbar))*(ground_HF_H + mu_B*b*(g_I*Iz + g_J_S1*Jz))

    # Find eigenvalues and eigenvectors
    eig_val, eig_vec = np.linalg.eig(ground_Z_H)
    energy.append(np.sort(np.real(eig_val))[::-1])
energy = np.array(energy).T


# P3/2
SJx, SJy, SJz = spin_mat(J_P2)
SIx, SIy, SIz = spin_mat(I)

Jx = np.kron(np.identity(int(2*I+1)), SJx)
Jy = np.kron(np.identity(int(2*I+1)), SJy)
Jz = np.kron(np.identity(int(2*I+1)), SJz)

Ix = np.kron(SIx, np.identity(int(2*J_P2 + 1)))
Iy = np.kron(SIy, np.identity(int(2*J_P2 + 1)))
Iz = np.kron(SIz, np.identity(int(2*J_P2 + 1)))


energy2 = []

# Hyperfine Hamiltonian for P3/2
P2_HF_H = A_P2*(np.matmul(Ix, Jx) + np.matmul(Iy, Jy) + np.matmul(Iz,Jz)) + (B_P2/(I*(2*I-1)*2*J_P2*(2*J_P2 - 1)))*(3*np.linalg.matrix_power(np.matmul(Ix, Jx) + np.matmul(Iy, Jy) + np.matmul(Iz,Jz), 2) + (3/2)*(np.matmul(Ix, Jx) + np.matmul(Iy, Jy) + np.matmul(Iz,Jz)) - I*(I+1)*J_P2*(J_P2 + 1)*np.identity(16))

# Now track both eigenvectors AND eigenvalues
energy2 = np.zeros((len(B), 16))
eigenvectors = np.zeros((len(B), 16, 16))

for i, b in enumerate(B):
    # Include Zeeman term
    P2_Z_H = (Amp / (2 * np.pi * hbar)) * (P2_HF_H + mu_B * b * (g_I * Iz + g_J_P2 * Jz))

    # Compute eigenvalues and eigenvectors
    eigvals, eig_vec = np.linalg.eig(P2_Z_H)
    eigvals = np.real(eigvals)  # Ensure real eigenvalues

    if i == 0:
        # Store initial eigenvalues and eigenvectors
        energy2[i, :] = eigvals
        eigenvectors[i, :, :] = eig_vec
    else:
        # Compute overlap matrix with previous step's eigenvectors
        # The overlap matrix is B^tA, where A contains all the eigenvectors from the current step,
        # and B contains all the eigenvectors from the previous step. Multiplying B^t by A essentially gives
        # a dot product of each eigenvector pairing. The largest dot product is the pairing that best maintains
        # continuity. This is all in order to track each mf level as it mixes and crosses other mf levels in the
        # intermediate Zeeman region.

        overlap = np.abs(np.dot(eigenvectors[i-1].T, eig_vec))

        # Find best-matching eigenvectors using maximum overlap
        indices = np.argmax(overlap, axis=1)

        # Reorder eigenvalues and eigenvectors to maintain continuity
        energy2[i, :] = eigvals[indices]
        eigenvectors[i, :, :] = eig_vec[:, indices]

energy2 = np.array(energy2).T

F3 = []
F2 = []
F1 = []
F0 = []

# Assign each eigenvalue to the appropriate energy level by looking at it's starting position
for i in range(16):
    if np.floor(energy2[i, 0]) == 193:
        F3.append(energy2[i, :])
    if np.ceil(energy2[i, 0]) == -72:
        F2.append(energy2[i, :])
    if np.ceil(energy2[i, 0]) == -229:
        F1.append(energy2[i, :])
    if np.ceil(energy2[i, 0]) == -302:
        F0.append(energy2[i, :])


col = plt.cm.inferno(np.linspace(0, 1,16))

fig, ax = plt.subplots(figsize=(10, 5))

# Plot each energy level

colors = ['#f6511d' ,'#ffb400', '#00a6ed', '#7fb800']
for i in range(2*3+1):
    ax.plot(B, F3[i], color = colors[0])
for i in range(2*2+1):
    ax.plot(B, F2[i], color = colors[1])
for i in range(2*1+1):
    ax.plot(B, F1[i], color = colors[2])
for i in range(2*0+1):
    ax.plot(B, F0[i], color = colors[3])

ax.text(s = 'F = 3', x = 0, y = 193 + 100)
ax.text(s = 'F = 2', x = 0, y = -72 + 60)
ax.text(s = 'F = 1', x = 0, y = -229 + 30)
ax.text(s = 'F = 0', x = 0, y = -302 - 130)

ax.text(s = r'$m_j = 3/2$', x = 470, y = 1000)
ax.text(s = r'$m_j = 1/2$', x = 470, y = 600)
ax.text(s = r'$m_j = -1/2$', x = 470, y = -320)
ax.text(s = r'$m_j = -3/2$', x = 470, y = -1100)

ax.set_xlabel('B (G)')
ax.set_ylabel('E (Mhz)')
plt.show()