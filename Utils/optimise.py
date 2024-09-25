import numpy as np

def skills(skill_df):
    """
    :param skill_df: df of marks with skills used and thresholds
    :return: list of skills and a nested dictionary where each mark has a dictionary of skills and their thresholds
    """
    skill_dict = {} #The skills used by each mark and their thresholds
    all_skills = set() #Set of all skills used
    for m in list(dict.fromkeys(skill_df['Mark'].tolist())):
        skill_dict[m] = {}
        for ele in skill_df.loc[skill_df['Mark'] == m]['Skills'].values[0].split(','):
            print(ele)
        skills = [int(ele) for ele in skill_df.loc[skill_df['Mark'] == m]['Skills'].values[0].split(',')]
        for index, s in enumerate(skills):
            skill_dict[m][s] = float(skill_df.loc[skill_df['Mark'] == m]['T' + str(index+1)].values[0])
            all_skills.add(s)
    all_skills = sorted(list(all_skills))
    return all_skills, skill_dict

def student_scores(score_df, student):
    """
    :param score_df: df containing the scores for each mark for each student
    :param student: student identifier
    :return: array (binary) of student scores
    """
    s_scores = score_df[str(student)].to_numpy(dtype=int) #array of scores for marks
    return s_scores

def student_sigs(score_df, mark_dict, student):
    """
    :param score_df: df containing the scores for each mark for each student
    :param mark_dict: dictionary where keys are the questions and values are lists containing the marks belonging to that question
    :param student: student identifier
    :return: array (binary) of signals to use for the student
    """
    s_scores = student_scores(score_df, student) #array of scores for marks
    s_signals = np.zeros(len(s_scores), dtype=int) #array of signals for marks
    for q in mark_dict.keys():
        for index, mark in enumerate(mark_dict[q]):
            if index == 0:
                s_signals[mark-1] = 1
            else:
                if s_scores[mark - 2] == 1:
                    s_signals[mark-1] = 1
    return s_signals

def thresharray(dict, all_skills):
    """
    :param dict: skilldict containing skills and accompanying thresholds for each j
    :param all_skills: all skills used
    :return: N_skills * N_j array of skill thresholds for each j
    """
    thresharray = []
    for j, ts in dict.items():
        tj = [0 for s in all_skills]
        for skill, t in ts.items():
            tj[all_skills.index(skill)] = t
        thresharray.append(tj)
    return np.array(thresharray)

def Combs(tarray, all_skills):
    """
    :param tarray: an N_skills*N_j array of thresholds (0 if skills is not used)
    :param all_skills: all skills used
    :return: Transform thresholds into a list of combinations of skill levels (bins) which will be the same across all i.
    """
    lists = []
    for index, s in enumerate(all_skills):
        lists.append(sorted(list(set(tarray[:, index].tolist()))))  # remove duplicates and sort thresholds by size
    combs = list(itertools.product(*lists))
    return np.array(combs)

def Cond_PScore_setup(combs, thresharray, theta, all_skills):
    """
    :param combs: array of all possible skill combinations
    :param thresharray: array of skill thresholds for each mark j
    :param theta: array of parameters, each of question j
    :param all_skills: all skills used
    :return: For each comb, calculate whether this is competent for mark j given thresholds defined in thresharray.
        Then calculate the probability (conditional on comb) of achieving scores of 1 and 0 (Comp1, Comp0) for every j - this depends only on theta
    """
    Competent = []
    for j in thresharray:
        TFj = np.ones(combs.shape)
        TFj[np.where(combs < j[np.newaxis:, ])] = 0  # Replace 1s with 0s if skill level is insufficient
        Competentj = np.ones(combs.shape[0])
        for i in range(len(all_skills)):
            Competentj *= TFj[:, i]
        Competent.append(Competentj)
    Competent = np.stack(Competent, axis=1)  # N_combs * N_j array. Asks: "is comb c competent for mark j?"
    Comp0 = np.add(Competent * (1 - theta)[np.newaxis:, ], (1 - Competent) * theta[np.newaxis:, ])
    Comp1 = np.add(Competent * theta[np.newaxis:, ], (1 - Competent) * (1 - theta)[np.newaxis:, ])
    return Competent, Comp1, Comp0


def Cond_PScores(scores, sig, Comp1, Comp0):
    """
    :param scores: array of scores for i
    :param sig: array of sig for i
    :param Comp1: array defined in prev function
    :param Comp0: array defined in prev function
    :return: Yield P(Scores|Comb) = P(Score_i1|comb)*P(Score_i2|comb)*...*P(Score_iNj|comb) for only the signals (j) we want to keep, using i's actual scores.
    """
    Cond = np.add(Comp1 * scores[np.newaxis:, ],
                  Comp0 * np.array(1 - scores)[np.newaxis:, ])  # return an array of P(Score_j | Comb) (for all j)
    mask = sig == 1
    Cond = Cond[:, mask]  # removing scores for j we do not want in the model
    Ones = np.ones(Cond.shape[0])
    for i in range(Cond.shape[1]):
        Ones *= Cond[:, i]  # Multiply columns in array together yielding P(Scores|Comb) (for each comb).
    return Ones

def Priors(arr, thresh):
    """
    :param arr: array reflecting the prior distribution of skill s (from dict priors)
    :param thresh: list of thresholds for skill s
    :return: Takes the new threshold values and finds the probability of the skill s falling in bins defined by thresholds
        according to prior defined by arr. Uses P(s<x) = Cumulative + ((x-upper)/(upper-lower))*P(lower < s < upper).
    """
    thresh.append(0)  # Zero must be a threshold value since some j won't use s
    thresh = sorted(set(thresh))
    Levels = arr[:, 0].tolist()
    Levels.append(1)
    bins = []
    cump = 0  # Cumulative Probability (missing out current)
    for i in range(len(Levels) - 1):
        p = float(arr[i, 1])
        bin = [Levels[i], Levels[i + 1], p, cump]
        cump += p
        bins.append(bin)
    bins = np.array(bins)
    categorised = []
    for t in thresh:
        mask = (t >= bins[:, 0]) & (t < bins[:, 1])  # Sort new thresholds into bins
        lst = bins[mask, :][0].tolist()
        lst.insert(0, t)
        categorised.append(lst)
    categorised = np.array(categorised)
    CDF = categorised[:, 4] + np.multiply(
        (categorised[:, 0] - categorised[:, 1]) / (categorised[:, 2] - categorised[:, 1]),
        categorised[:, 3])  # Calculate probability
    CDFList = CDF.tolist()
    CDFList.append(1)
    pri = []
    for i in range(len(thresh)):
        cum = [thresh[i]]
        cum.append(CDFList[i + 1] - CDFList[i])
        pri.append(cum)
    return np.array(pri)



def PSkills(combs, priors, thresharray, all_skills):
    """
    :param combs: array of different combinations of skills
    :param priors: a list of arrays - each dictates the prior distribution over s.
    :param thresharray: array of all thresholds used in all questions. If s is not used in j, thresh = 0. (N_j*N_s array)
    :param all_skills: all skills used
    :return: Here we take each combination and calculate the probability of seeing each skill level in this comb conditional on prior distribution.
        We then multiply the probabilities together to obtain P(skills) for each comb then multiply by P(cond).
        We also return an array for each skill denoting the bins that each skill value falls into.
    """
    p = np.ones(combs.shape[0])
    sarrays = []
    for i in range(len(all_skills)):
        Pri = Priors(priors[i], thresharray[:, i].tolist())
        col = combs[:, i]  # select comb column with skill i-1
        sarray = np.zeros(
            (col.shape[0], Pri.shape[0]))  # array of 1s and 0s indicating which bin t falls into for each s:
        pcol = np.ones(col.shape[0])
        for index, [t, pr] in enumerate(Pri):
            mask = col == t
            pcol[mask] = pr
            sarray[:, index][mask] = 1
        p *= pcol  # Multiply cols together
        sarrays.append(sarray)
    finp = p
    return finp, sarrays


def Posteriors(finp, sarrays, thresharray, condp, all_skills):
    """
    :param combined: array containing each combination, the probability of achieving observed scores conditional on comb and the prior probability of observing this comb
    :param sarrays: list of arrays. Array for each skill reflecting the bins that each skill value falls into
    :param thresharray: array of all thresholds used in all questions. If s is not used in j, thresh = 0. (N_j*N_s array)
    :param condp: X
    :param all_skills: all skills used
    :return: We would like to apply the bayesian updating routine to each bin for each skill and return output as an array of posteriors for each skill
    """
    finp = finp * condp
    P_total = np.sum(finp)  # Sum_y(P(score|y,theta_t)P(y|theta_t) (prior -> uniform)
    Posterior = []
    for i in range(len(all_skills)):
        threshlist = sorted(set(thresharray[:,
                                i].tolist()))  # This is the way threshs are organised in previous function (see Priors())
        sarray = sarrays[i]
        Posterior_s = []
        for t in range(len(threshlist)):
            p = finp[sarray[:, t] == 1]  # keep values in finp s.t. this is threshold
            Posterior_s.append([threshlist[t], np.sum(p)])
        Posterior_s = np.array(Posterior_s)
        Posterior_s[:, 1] *= 1 / P_total
        Posterior.append(Posterior_s)
    return Posterior, P_total


def PComp(Competent, finp, sig, scores):
    """
    :param Competent: N_j*N_combs boolean array telling us if comb is competent for each j
    :param finp: N_combs array: Prior P(Comb) for all combs (will be posteriors in e step)
    :param sig: X
    :param scores: X
    :return: For each j, find the P(competent) (Sum P(Combs) for where comb is competent
    """
    mask = sig == 1
    scores = scores[mask]
    Competent = Competent[:, mask]
    pskills = finp
    pcomp = Competent * pskills[:, None]
    pcomp = pcomp.sum(axis=0)
    pcomp = (scores * pcomp) + ((1 - scores) * (1 - pcomp))  # if score=1, return pcomp; if score=0, return 1-comp
    pcomp_array = []
    index = 0
    for j in range(sig.shape[0]):
        if sig[j] == 1:
            pcomp_array.append(float(pcomp[index]))
            index += 1
        else:
            pcomp_array.append(0)
    return np.array(pcomp_array)

def e_step(theta, pcombs, combs, tarray, sarray, scores, signals, all_skills):
    Co, C1, C0 = Cond_PScore_setup(combs, tarray, theta, all_skills)
    pcomp = {}
    posteriors = {}
    logl = 0
    for i in scores:
        scores_i = scores[i]
        signals_i = signals[i]
        Cond_P = Cond_PScores(scores_i, signals_i, C1, C0)
        post, p = Posteriors(pcombs, sarray, tarray, Cond_P, all_skills)
        posteriors[i] = post
        logl_i = -np.log(p)
        logl += logl_i
        finp_post = PSkills(combs, post, tarray, all_skills)[0]
        pcomp[i] = PComp(Co, finp_post, signals_i, scores_i)

    return posteriors, pcomp, logl


def m_step(pcomp, signals, mlist):
    """
    :param pcomp: dictionary of pcomp (posterior probability of having skills that are competent for each j) for each i
    :param signals: dictionary of signals for each i
    :param mlist: list of marks we are running model on
    :returns: Parameter updates on theta
    """

    sig_array = []
    pcomp_removed = np.zeros(len(mlist))
    for i, array in signals.items():
        sig_array.append(array)
        pcomp_removed += pcomp[i] * array
    N_sig = np.array(sig_array).sum(axis=0)

    theta_new = list(np.zeros(len(mlist)))
    for j in range(len(mlist)):
        if N_sig[j] != 0:
            theta_new[j] = pcomp_removed[j] / N_sig[j]
        else:
            theta_new[j] = 0.5
    return np.array(theta_new)


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
