import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on BIC scores
        def bic(logL, p, logN):
            return -2*logL + p*logN

        min_score = float('inf')
        best_model = None
        n_features = self.X.shape[1]
        N = len(self.X)
        logN = math.log(N)
        for n_comp in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_comp)
                logL = model.score(self.X, self.lengths)
                # Dana Sheahen on slack: p = n^2 + 2*d*n - 1
                p = n_comp**2 + 2*n_comp*n_features - 1
                score = bic(logL, p, logN)
                if score < min_score:
                    min_score = score
                    best_model = model
            except:
                continue
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores
        def dic(logLs, M):
            logL_i = logLs[self.this_word]
            LogLs_allbuti = [logLs[word] for word in self.words if word != 
                             self.this_word]
            return logL_i - sum(LogLs_allbuti)/(M - 1)

        best_score = float('-inf')
        best_model = None
        M = len(self.words)
        for n_comp in range(self.min_n_components, self.max_n_components + 1):
                try:
                    logLs = {}
                    model = self.base_model(n_comp)
                    for word in self.words:
                        X, lengths = self.hwords[word]
                        logLs[word] = model.score(X, lengths)
                    score = dic(logLs, M)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except:
                    continue
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection using CV
        best_model = None
        best_score = float('-inf')
        len_seq = len(self.sequences)
        if len_seq == 1:
            return best_model

        n_splits = min(3, len_seq)
        kf = KFold(n_splits, random_state=self.random_state)

        for n_comp in range(self.min_n_components, self.max_n_components + 1):
            logLs = []
            try:
                for train_idx, test_idx in kf.split(self.sequences):
                    X_train, lengths_train = combine_sequences(
                        train_idx, 
                        self.sequences
                    )
                    X_test, lengths_test = combine_sequences(
                        test_idx, 
                        self.sequences
                    )
                    model = GaussianHMM(
                        n_components=n_comp, 
                        covariance_type="diag", 
                        n_iter=1000,
                        random_state=self.random_state, 
                        verbose=False
                    ).fit(X_train, lengths_train)
                    logLs.append(model.score(X_test, lengths_test))
                score = np.mean(logLs)
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                continue
        return best_model
