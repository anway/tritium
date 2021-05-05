import argparse
import math
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg
import timeit
from timeit import default_timer

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nu", help="momenta in range [-nu, nu]^3")
args = parser.parse_args()

nu_max = int(args.nu)
D = 10.
omega = (2 * D) ** 3
nuclei_pos = [[0., 0., 0.], [0., 0., 1.398]]
nuclei_q = [1., 1.]

N = nu_max ** 3

ks = []
rs = []
for nu_x in range(-1 * nu_max, nu_max):
   for nu_y in range(-1 * nu_max, nu_max):
      for nu_z in range(-1 * nu_max, nu_max):
         nu = np.array([nu_x, nu_y, nu_z])
         ks.append(2. * math.pi * nu / (omega ** (1. / 3.)))
         rs.append(nu * (omega / (8. * N)) ** (1. / 3.))

T_mat = sparse.dok_matrix((64 * N * N, 64 * N * N))
UV_mat = sparse.dok_matrix((64 * N * N, 64 * N * N))
for i_1, k_1 in enumerate(ks):
   for i_2, k_2 in enumerate(ks):
      i = i_1 * 8 * N + i_2

      T_mat[i, i] = 0.5 * (np.dot(k_1, k_1) + np.dot(k_2, k_2))

      U_1 = -4. * math.pi / omega * \
            sum([charge * np.cos(np.dot(k_nu, pos - rs[i_1])) / np.dot(k_nu, k_nu) \
               for charge, pos in zip(nuclei_q, nuclei_pos) for k_nu in ks if \
               not np.array_equal(k_nu, np.array([0., 0., 0.])) and \
               np.linalg.norm(pos - rs[i_1]) <= D])
      U_2 = -4. * math.pi / omega * \
            sum([charge * np.cos(np.dot(k_nu, pos - rs[i_2])) / np.dot(k_nu, k_nu) \
               for charge, pos in zip(nuclei_q, nuclei_pos) for k_nu in ks if \
               not np.array_equal(k_nu, np.array([0., 0., 0.])) and \
               np.linalg.norm(pos - rs[i_2]) <= D])
      V = 2. * math.pi / omega * \
            sum([np.cos(np.dot(k_nu, rs[i_1] - rs[i_2])) / np.dot(k_nu, k_nu) for k_nu in ks if \
               not np.array_equal(k_nu, np.array([0., 0., 0.])) and \
               np.linalg.norm(rs[i_1] - rs[i_2]) <= D])
      UV_mat[i, i] = U_1 + U_2 + V

# nu to p
def dft(psi):
   psis = np.reshape(psi, (2 * nu_max, 2 * nu_max, 2 * nu_max, 2 * nu_max, 2 * nu_max, 2 * nu_max))
   psis = 1. / N * np.fft.fftshift(np.fft.fftn(np.fft.fftshift(psis)))
   return np.reshape(psis, (64 * N * N, ))

# p to nu
def idft(psi):
   psis = np.reshape(psi, (2 * nu_max, 2 * nu_max, 2 * nu_max, 2 * nu_max, 2 * nu_max, 2 * nu_max))
   psis =  N * np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(psis)))
   return np.reshape(psis, (64 * N * N, ))

# Matrix vector multiply
def mv(psi):
   psiT = T_mat.dot(psi)
   psiU = dft(psi)
   psiU = UV_mat.dot(psiU)
   psiU = idft(psiU)
   return psiT + psiU

H = sparse.linalg.LinearOperator((64 * N * N, 64 * N * N), matvec=mv)
print("diagonalizing", flush=True)
start_time = timeit.default_timer()
evals, evecs = sparse.linalg.eigsh(H, which='SA')
tot_time = timeit.default_timer() - start_time
print(evals)
print(tot_time)
