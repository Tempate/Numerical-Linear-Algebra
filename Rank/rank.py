import scipy.linalg as sl


TOLERANCE = 1e-12


def rank(matrix, method):
    if method == "svd":
        _, s, _ = sl.svd(matrix)
        return count_nonzero(s)

    if method == "qr":
        _, r = sl.qr(matrix)
        return count_nonzero(r.diagonal())

    if method == "lu":
        lu = sl.lu_factor(matrix)[0]
        return count_nonzero(lu.diagonal())

    raise Exception("Invalid method") 


def count_nonzero(vector):
    is_not_zero = lambda element: abs(element) > TOLERANCE

    return sum(int(is_not_zero(element)) for element in vector)
