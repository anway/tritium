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
omega = 10.
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
         rs.append(nu * ((omega / N)) ** (1. / 3.))

rk_prod = []
for r in rs:
   row = []
   for k in ks:
      row.append(np.dot(r, k))
   rk_prod.append(row)
rk_prod_arr = np.array(rk_prod)

# nu to p
def dft(psis):
   exps = np.exp(-1j * np.add.outer(rk_prod_arr, rk_prod_arr))
   return 1. / N * np.tensordot(psis, exps, axes=([0, 1], [1, 3]))

# p to nu
def idft(psis):
   exps = np.exp(1j * np.add.outer(rk_prod_arr, rk_prod_arr))
   return 1. / (64. * N) * np.tensordot(psis, exps, axes=([0, 1], [0, 2]))

# Matrix vector multiply
def mv(psi):
   psis = np.reshape(psi, (8 * N, 8 * N))
   # T term
   E_T = 0.
   for i_1, i_2 in np.ndindex(psis.shape):
      E_T += 1. / 2. * (np.dot(ks[i_1], ks[i_1]) + np.dot(ks[i_2], ks[i_2]))   
   psis[i_1, i_2] *= E_T   
   psis = dft(psis)
   # U term
   E_U = 0.
   for i_1, i_2 in np.ndindex(psis.shape):
      U_1 = -4. * math.pi / omega * \
            sum([charge * np.cos(np.dot(k_nu, pos - rs[i_1])) / np.dot(k_nu, k_nu) \
               for charge, pos in zip(nuclei_q, nuclei_pos) for k_nu in ks if \
               not np.array_equal(k_nu, np.array([0., 0., 0.]))])
      U_2 = -4. * math.pi / omega * \
            sum([charge * np.cos(np.dot(k_nu, pos - rs[i_2])) / np.dot(k_nu, k_nu) \
               for charge, pos in zip(nuclei_q, nuclei_pos) for k_nu in ks if \
               not np.array_equal(k_nu, np.array([0., 0., 0.]))])
      E_U += U_1 + U_2
   # V term
   E_V = 0.
   for i_1, i_2 in np.ndindex(psis.shape):
      r = rs[i_2] - rs[i_1]
      V = 2. * math.pi / omega * \
            sum([np.cos(np.dot(k_nu, r)) / np.dot(k_nu, k_nu) for k_nu in ks if \
               not np.array_equal(k_nu, np.array([0., 0., 0.]))])
      E_V += V
   psis[i_1, i_2] *= (E_U + E_V)
   psis = idft(psis)
   return np.reshape(psis, (64 * N * N, 1))

H = sparse.linalg.LinearOperator((64 * N * N, 64 * N * N), matvec=mv)
print("diagonalizing", flush=True)
evals, evecs = sparse.linalg.eigsh(H, which='SA')
print(evals)
