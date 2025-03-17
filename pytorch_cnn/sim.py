# Simulate single asset using GBM
import numpy as np
from scipy.stats import bernoulli


class BS_asset_sim:
    def __init__(self, miu, sigma, rho, J_apt, _lambda, S_0, T, t, time_step, N):
        self.miu = miu
        self.sigma = sigma
        self.rho = rho
        self.J_apt = J_apt
        self._lambda = _lambda
        self.S_0 = S_0
        self.T = T
        self.t = t
        self.time_step = time_step
        self.N = N

    def dt(self):
        return 1 / self.time_step

    @staticmethod
    def sigma_sum_sq(sigma):
        return np.sum(sigma ** 2)

    @staticmethod
    def diffusion_sum(sigma, rho, n_path, N):
        W_1 = np.random.normal(0, 1, (n_path, N))
        Z_2 = np.random.normal(0, 1, (n_path, N))
        W_2 = np.array([rho * x + np.sqrt(1 - rho ** 2) * y for x, y in zip(W_1, Z_2)])
        return sigma[0] * W_1 + sigma[1] * W_2

    def exact(self):
        miu = self.miu
        sigma = self.sigma
        rho = self.rho
        J_apt = self.J_apt
        _lambda = self._lambda
        S_0 = self.S_0
        T = self.T
        t = self.t
        time_step = self.time_step
        N = self.N

        n_path = round((T - t) * time_step) + 1
        S_array = np.zeros((N, n_path))
        S_array[:, 0] = np.log(S_0)
        sigma_sum_sq = BS_asset_sim.sigma_sum_sq(sigma)
        np.random.seed(10)
        diffusion_sum = BS_asset_sim.diffusion_sum(sigma, rho, n_path, N)
        np.random.seed(1)
        N_t = bernoulli.rvs(size=(n_path, N), p=1 - np.exp(-_lambda * self.dt()))
        np.random.seed(5)
        J = np.random.uniform(J_apt[0], J_apt[1], (n_path, N))
        for i in range(1, n_path):
            S_array[:, i] = S_array[:, i - 1] + (
                        miu - (1 / 2) * (sigma_sum_sq + 2 * np.prod(sigma) * rho)) * self.dt() + diffusion_sum[
                                i] * self.dt() ** (1 / 2) + N_t[i] * J[i]
        return np.exp(S_array)


if __name__ == "__main__":
    time_step = 252
    N = 1000
    obj = BS_asset_sim(miu=0.06, sigma=np.array([0.2, 0.3]), rho=-0.2, J_apt=[-0.02, 0.01], _lambda=5, S_0=400, T=1,
                       t=0, time_step=time_step, N=N)
    path = obj.exact().mean(axis=0)
    path_sim = obj.exact()
    print(path)
    ret = np.prod(1 + np.diff(path) / path[:-1]) - 1
    vol = np.std(np.diff(path) / path[:-1]) * time_step ** 0.5
    # print(ret, vol)