import pytest
import numpy as np
from genetic_algorithm.genetic_algorithm_scripts.analysis import make_cv_sets, get_dict_key


class TestClass:

    def test_make_cv_sets(self):
        X = []
        y = [i for i in range(10)]

        for i in range(10):
            X.append([j for j in range(10)])

        X = np.array(X)
        y = np.array(y)

        window = 1
        cv = 1

        X_train, X_test, y_train, y_test = make_cv_sets(X, y, window, cv)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) == 1

        cv = 5
        X_train, X_test, y_train, y_test = make_cv_sets(X, y, window, cv)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) == 5

        cv = 10
        X_train, X_test, y_train, y_test = make_cv_sets(X, y, window, cv)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) == 10

    def test_get_dict_key(self):

        keys = ["a", "b", "c"]
        dictionary = {"a": 1, "b": 2, "c": 3}

        for i in range(len(dictionary)):
            key = get_dict_key(dictionary, i)
            assert key == keys[i]
            

if __name__=="__main__":
    tester = TestClass()
    tester.test_make_cv_sets()
    tester.test_get_dict_key()