import argparse
import math
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nu", help="momenta in range [-nu, nu]^3")
args = parser.parse_args()

nu_max = int(args.nu)
D = 10.
omega = (2 * D) ** 3
nuclei_pos = [[0., 0., 0.]]
nuclei_q = [1.]

N = nu_max ** 3

ks = []
rs = []
for nu_x in range(-1 * nu_max, nu_max):
   for nu_y in range(-1 * nu_max, nu_max):
      for nu_z in range(-1 * nu_max, nu_max):
         nu = np.array([nu_x, nu_y, nu_z])
         ks.append(2. * math.pi * nu / (omega ** (1. / 3.)))
         rs.append(nu * (omega / (2. * N)) ** (1. / 3.))

T_mat = sparse.dok_matrix((8 * N, 8 * N))
UV_mat = sparse.dok_matrix((8 * N, 8 * N))
for i_1, k_1 in enumerate(ks):
   T_mat[i_1, i_1] = 0.5 * np.dot(k_1, k_1)
   U_1 = -4. * math.pi / omega * \
         sum([charge * np.cos(np.dot(k_nu, pos - rs[i_1])) / np.dot(k_nu, k_nu) \
            for charge, pos in zip(nuclei_q, nuclei_pos) for k_nu in ks if \
            not np.array_equal(k_nu, np.array([0., 0., 0.])) and \
            np.linalg.norm(pos - rs[i_1]) <= D])
   UV_mat[i_1, i_1] = U_1

# nu to p
def dft(psi):
   psis = np.reshape(psi, (2 * nu_max, 2 * nu_max, 2 * nu_max))
   psis = 1. / N * np.fft.fftshift(np.fft.fftn(psis))
   return np.reshape(psis, (8 * N, 1))

# p to nu
def idft(psi):
   psis = np.reshape(psi, (2 * nu_max, 2 * nu_max, 2 * nu_max))
   psis =  N * np.fft.ifftshift(np.fft.ifftn(psis))
   return np.reshape(psis, (8 * N, 1))

# Matrix vector multiply
def mv(psi):
   psis = T_mat.dot(psi)
   psis = dft(psis)
   psis = UV_mat.dot(psis)
   psis = idft(psis)
   return psis

H = sparse.linalg.LinearOperator((8 * N, 8 * N), matvec=mv)
print("diagonalizing", flush=True)
evals, evecs = sparse.linalg.eigsh(H, which='SA')
print(evals)
