import numpy as np

### Question 1 ###

# Initial data

x_1 = np.array([[1], [2]])
x_2 = np.array([[-1], [1]])
x_3 = np.array([[1], [0]])

mu_1 = np.array([[2], [2]])
mu_2 = np.array([[0], [0]])

Sigma_1 = np.array([[2, 1], [1, 2]])
Sigma_2 = np.array([[2, 0], [0, 2]])

from scipy.stats import multivariate_normal


def calc_likelihoods(x, mu_1, mu_2, Sigma_1, Sigma_2):
    likelihood_1 = multivariate_normal(mu_1, Sigma_1).pdf(x.T)
    likelihood_2 = multivariate_normal(mu_2, Sigma_2).pdf(x.T)
    return np.array([likelihood_1, likelihood_2])


def calc_posteriors(priors, likelihoods):
    posteriors = np.array([])
    for i in range(len(priors)):
        posteriors = np.append(posteriors, priors[i] * likelihoods[i])

    return posteriors / np.sum(posteriors)  # normalize


def update_means(k1_posteriors, k2_posteriors):
    mu_1 = np.zeros((2, 1), dtype=float)
    mu_2 = np.zeros((2, 1), dtype=float)

    for i in range(len(k1_posteriors)):
        x = eval(f"x_{i+1}")
        mu_1 += k1_posteriors[i] * x
        mu_2 += k2_posteriors[i] * x

    return mu_1 / np.sum(k1_posteriors), mu_2 / np.sum(k2_posteriors)


def update_covs(k1_posteriors, k2_posteriors, mu_1, mu_2):
    Sigma_1 = np.zeros((2, 2), dtype=float)
    Sigma_2 = np.zeros((2, 2), dtype=float)

    for i in range(len(k1_posteriors)):
        x = eval(f"x_{i+1}")
        Sigma_1 += k1_posteriors[i] * (x - mu_1) @ (x - mu_1).T
        Sigma_2 += k2_posteriors[i] * (x - mu_2) @ (x - mu_2).T

    return Sigma_1 / np.sum(k1_posteriors), Sigma_2 / np.sum(k2_posteriors)


def update_priors(k1_posteriors, k2_posteriors):
    total = np.sum(k1_posteriors) + np.sum(k2_posteriors)
    return np.sum(k1_posteriors) / total, np.sum(k2_posteriors) / total


mu_1_vector = mu_1.transpose()[0]
mu_2_vector = mu_2.transpose()[0]

priors = np.array([0.5, 0.5])

p_x_1_given_k_1, p_x_1_given_k_2 = calc_likelihoods(
    x_1, mu_1_vector, mu_2_vector, Sigma_1, Sigma_2
)
p_x_2_given_k_1, p_x_2_given_k_2 = calc_likelihoods(
    x_2, mu_1_vector, mu_2_vector, Sigma_1, Sigma_2
)
p_x_3_given_k_1, p_x_3_given_k_2 = calc_likelihoods(
    x_3, mu_1_vector, mu_2_vector, Sigma_1, Sigma_2
)

posteriors_x_1 = calc_posteriors(priors, [p_x_1_given_k_1, p_x_1_given_k_2])
posteriors_x_2 = calc_posteriors(priors, [p_x_2_given_k_1, p_x_2_given_k_2])
posteriors_x_3 = calc_posteriors(priors, [p_x_3_given_k_1, p_x_3_given_k_2])

k1_posteriors = np.array([posteriors_x_1[0], posteriors_x_2[0], posteriors_x_3[0]])
k2_posteriors = np.array([posteriors_x_1[1], posteriors_x_2[1], posteriors_x_3[1]])

# update the parameters

mu_1_after_update, mu_2_after_update = update_means(k1_posteriors, k2_posteriors)
Sigma_1_after_update, Sigma_2_after_update = update_covs(
    k1_posteriors, k2_posteriors, mu_1_after_update, mu_2_after_update
)

# update the priors

priors_after_update = update_priors(k1_posteriors, k2_posteriors)
prior_1_update, prior_2_update = priors_after_update

### Question 2a ###

mu_1_after_update_vector = mu_1_after_update.transpose()[0]
mu_2_after_update_vector = mu_2_after_update.transpose()[0]

updated_p_x_1_given_k_1, updated_p_x_1_given_k_2 = calc_likelihoods(
    x_1,
    mu_1_after_update_vector,
    mu_2_after_update_vector,
    Sigma_1_after_update,
    Sigma_2_after_update,
)
updated_p_x_2_given_k_1, updated_p_x_2_given_k_2 = calc_likelihoods(
    x_2,
    mu_1_after_update_vector,
    mu_2_after_update_vector,
    Sigma_1_after_update,
    Sigma_2_after_update,
)
updated_p_x_3_given_k_1, updated_p_x_3_given_k_2 = calc_likelihoods(
    x_3,
    mu_1_after_update_vector,
    mu_2_after_update_vector,
    Sigma_1_after_update,
    Sigma_2_after_update,
)

updated_posteriors_x_1 = calc_posteriors(
    priors_after_update, [updated_p_x_1_given_k_1, updated_p_x_1_given_k_2]
)
updated_posteriors_x_2 = calc_posteriors(
    priors_after_update, [updated_p_x_2_given_k_1, updated_p_x_2_given_k_2]
)
updated_posteriors_x_3 = calc_posteriors(
    priors_after_update, [updated_p_x_3_given_k_1, updated_p_x_3_given_k_2]
)

hard_assignments = (
    np.argmax(
        np.array(
            [updated_posteriors_x_1, updated_posteriors_x_2, updated_posteriors_x_3]
        ),
        axis=1,
    )
    + 1
)

### Question 2b ###

# calculate the norm between x1 and x2, x2 and x3, x1 and x3

norm_x1_x2 = np.linalg.norm(x_1 - x_2)
norm_x2_x3 = np.linalg.norm(x_2 - x_3)
norm_x1_x3 = np.linalg.norm(x_1 - x_3)

# calculate the silhouette

s_2 = (norm_x1_x2 - norm_x2_x3) / norm_x1_x2
s_3 = (norm_x1_x3 - norm_x2_x3) / max(norm_x1_x3, norm_x2_x3)

s_k_2 = (s_2 + s_3) / 2
