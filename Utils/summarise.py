def plimit(thersharray, posterior, allskills, theta, scores, sigs):
    """
    :param thresharray:
    :param posteriors:
    :returns lim_array: an array with width N_skills reflecting P(Score = 0 & Skill<Threshold) for each incorrect mark used as a signal.
    """

    mask = (scores == 0) & (sigs == 1)  # Only use signals where score = 0 and sig = 1
    tarray = thersharray[mask]
    thet = theta[mask]
    skill_usage = np.sum(tarray != 0, axis=0)
    lim_array = np.zeros(tarray.shape)
    for s in range(len(allskills)):
        pos_s = posterior[s]
        probs = pos_s[:, 1]
        probs_adjusted = np.insert(probs[:-1], 0, 0)
        pos_s_cdf = np.array([pos_s[:, 0], np.cumsum(probs_adjusted)]).T
        cdf_dict = {key: value for key, value in pos_s_cdf}
        lim_array[:, s] = np.array(
            [cdf_dict.get(item, item) for item in tarray[:, s]])  # ignores when thresh = 0 since p(s<0) = 0
        lim_array[:, s] = lim_array[:, s] * thet
    return lim_array, skill_usage

def get_expectations(posterior, allskills):
    """
    :param posterior: Given posterior distributions for an i for all skills
    :param allskills: list of all skills tested
    :return: expectations for each skill E(Y^k_i)
    """
    expectations = []
    for i in range(len(allskills)):
        pos_s = posterior[i]
        a = pos_s[:, 0]
        a = np.delete(a, 0)
        a = np.append(a, 1)
        means = (pos_s[:, 0] + a) / 2
        weighted_means = means * pos_s[:, 1]
        expectations.append(np.sum(weighted_means))
    expectations = np.array(expectations)
    return expectations
