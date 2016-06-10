from sympy.core.sympify import sympify
from sympy.sets import *
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.solvers.solveset import solveset, _has_rational_power
from sympy.simplify import simplify
from sympy import S, log, Pow
from sympy.series.limits import limit


def singularities(expr, sym):
    """
    Finds singularities for a function.
    Currently supported functions are:
    - univariate rational(real or complex) functions

    Examples
    ========

    >>> from sympy.calculus.singularities import singularities
    >>> from sympy import Symbol, I, sqrt
    >>> x = Symbol('x', real=True)
    >>> y = Symbol('y', real=False)
    >>> singularities(x**2 + x + 1, x)
    EmptySet()
    >>> singularities(1/(x + 1), x)
    {-1}
    >>> singularities(1/(y**2 + 1), y)
    {-I, I}
    >>> singularities(1/(y**3 + 1), y)
    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Mathematical_singularity

    """
    if not expr.is_rational_function(sym):
        raise NotImplementedError("Algorithms finding singularities for"
                                  " non rational functions are not yet"
                                  " implemented")
    else:
        return solveset(simplify(1/expr), sym)


def function_range(f, symbol, domain):
    """
    Finds the range of a function in a given domain.
    This method is limited by the ability to determine the singularities and
    determine limits.

    Examples
    ========

    >>> from sympy import Symbol, S, exp, log, pi, sqrt, sin, tan
    >>> from sympy.sets import Interval
    >>> from sympy.calculus.singularities import function_range
    >>> x = Symbol('x')
    >>> function_range(sin(x), x, Interval(0, 2*pi))
    [-1, 1]
    >>> function_range(tan(x), x, Interval(-pi/2, pi/2))
    (-oo, oo)
    >>> function_range(1/x, x, S.Reals)
    (-oo, oo)
    >>> function_range(exp(x), x, S.Reals)
    (0, oo)
    >>> function_range(log(x), x, S.Reals)
    (-oo, oo)
    >>> function_range(sqrt(x), x , Interval(-5, 9))
    [0, 3]

    """
    vals = S.EmptySet
    intervals = continuous_in(f, symbol, domain)
    range_int = S.EmptySet
    if isinstance(intervals, Interval):
        interval_iter = (intervals,)
    else:
        interval_iter = intervals.args

    for interval in interval_iter:
        cps = S.EmptySet
        cvs = S.EmptySet
        bounds = ((interval.left_open, interval.inf, '+'),
                  (interval.right_open, interval.sup, '-'))

        for i in bounds:
            if i[0]:
                cvs += FiniteSet(limit(f, symbol, i[1], i[2]))
                vals += cvs
            else:
                vals += FiniteSet(f.subs(symbol, i[1]))

        cps += solveset(f.diff(symbol), symbol, domain)

        for cp in cps:
            vals += FiniteSet(f.subs(symbol, cp))

        left_open, right_open = False, False

        if cvs is not S.EmptySet:
            if cvs.inf == vals.inf:
                left_open = True
            if cvs.sup == vals.sup:
                right_open = True

        range_int += Interval(vals.inf, vals.sup, left_open, right_open)

    return range_int


def continuous_in(f, symbol, interval):
    """
    Finds the intervals of continuity of a function in a given interval range.
    This method is limited by the ability to determine the various
    singularities and discontinuities of the given function.

    Examples
    ========
    >>> from sympy import Symbol, S, tan, log, pi, sqrt
    >>> from sympy.sets import Interval
    >>> from sympy.calculus.singularities import continuous_in
    >>> x = Symbol('x')
    >>> continuous_in(1/x, x, S.Reals)
    (-oo, 0) U (0, oo)
    >>> continuous_in(tan(x), x, Interval(0, pi))
    [0, pi/2) U (pi/2, pi]
    >>> continuous_in(sqrt(x - 2), x, Interval(-5, 5))
    [2, 5]
    >>> continuous_in(log(2*x - 1), x, S.Reals)
    (1/2, oo)

    """
    if interval.is_subset(S.Reals):
        constrained_interval = interval
        for atom in f.atoms(Pow):
            predicate, denom = _has_rational_power(atom, symbol)
            constraint = S.EmptySet
            if predicate and denom == 2:
                constraint = solve_univariate_inequality(atom.base >= 0,
                                                         symbol).as_set()
                constrained_interval = Intersection(constraint,
                                                    constrained_interval)
        for atom in f.atoms(log):
            constraint = solve_univariate_inequality(atom.args[0] > 0,
                                                     symbol).as_set()
            constrained_interval = Intersection(constraint,
                                                constrained_interval)
        interval = constrained_interval
    try:
        sings = S.EmptySet
        for atom in f.atoms(Pow):
            predicate, denom = _has_rational_power(atom, symbol)
            if predicate and denom == 2:
                sings = solveset(1/f, symbol, interval)
                break
        else:
            sings = Intersection(solveset(1/f, symbol), interval)
    except:
        raise NotImplementedError("Methods for determining the continuous domains"
                                  " of this function has not been developed.")

    return interval - sings


###########################################################################
###################### DIFFERENTIAL CALCULUS METHODS ######################
###########################################################################


def is_increasing(f, interval=S.Reals, symbol=None):
    """
    Returns if a function is increasing or not, in the given
    ``Interval``.

    Examples
    ========

    >>> from sympy import is_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_increasing(-x**2, Interval(-oo, 0))
    True
    >>> is_increasing(-x**2, Interval(0, oo))
    False
    >>> is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))
    False
    >>> is_increasing(x**2 + y, Interval(1, 2), x)
    True

    """
    f = sympify(f)
    free_sym = f.free_symbols

    if symbol is None:
        if len(free_sym) > 1:
            raise NotImplementedError('is_increasing has not yet been implemented '
                                        'for all multivariate expressions')
        if len(free_sym) == 0:
            return True
        symbol = free_sym.pop()

    df = f.diff(symbol)
    df_nonneg_interval = solveset(df >= 0, symbol, domain=S.Reals)
    return interval.is_subset(df_nonneg_interval)


def is_strictly_increasing(f, interval=S.Reals, symbol=None):
    """
    Returns if a function is strictly increasing or not, in the given
    ``Interval``.

    Examples
    ========

    >>> from sympy import is_strictly_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import Interval, oo
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3))
    False
    >>> is_strictly_increasing(-x**2, Interval(0, oo))
    False
    >>> is_strictly_increasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    f = sympify(f)
    free_sym = f.free_symbols

    if symbol is None:
        if len(free_sym) > 1:
            raise NotImplementedError('is_strictly_increasing has not yet been implemented '
                                        'for all multivariate expressions')
        elif len(free_sym) == 0:
            return False
        symbol = free_sym.pop()

    df = f.diff(symbol)
    df_pos_interval = solveset(df > 0, symbol, domain=S.Reals)
    return interval.is_subset(df_pos_interval)


def is_decreasing(f, interval=S.Reals, symbol=None):
    """
    Returns if a function is decreasing or not, in the given
    ``Interval``.

    Examples
    ========

    >>> from sympy import is_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    f = sympify(f)
    free_sym = f.free_symbols

    if symbol is None:
        if len(free_sym) > 1:
            raise NotImplementedError('is_decreasing has not yet been implemented '
                                        'for all multivariate expressions')
        elif len(free_sym) == 0:
            return True
        symbol = free_sym.pop()

    df = f.diff(symbol)
    df_nonpos_interval = solveset(df <= 0, symbol, domain=S.Reals)
    return interval.is_subset(df_nonpos_interval)


def is_strictly_decreasing(f, interval=S.Reals, symbol=None):
    """
    Returns if a function is strictly decreasing or not, in the given
    ``Interval``.

    Examples
    ========

    >>> from sympy import is_strictly_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_strictly_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_strictly_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    f = sympify(f)
    free_sym = f.free_symbols

    if symbol is None:
        if len(free_sym) > 1:
            raise NotImplementedError('is_strictly_decreasing has not yet been implemented '
                                        'for all multivariate expressions')
        elif len(free_sym) == 0:
            return False
        symbol = free_sym.pop()

    df = f.diff(symbol)
    df_neg_interval = solveset(df < 0, symbol, domain=S.Reals)
    return interval.is_subset(df_neg_interval)


def is_monotonic(f, interval=S.Reals, symbol=None):
    """
    Returns if a function is monotonic or not, in the given
    ``Interval``.

    Examples
    ========

    >>> from sympy import is_monotonic
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_monotonic(-x**2, S.Reals)
    False
    >>> is_monotonic(x**2 + y + 1, Interval(1, 2), x)
    True

    """
    from sympy.core.logic import fuzzy_or
    f = sympify(f)
    free_sym = f.free_symbols

    if symbol is None and len(free_sym) > 1:
        raise NotImplementedError('is_monotonic has not yet been '
                                'for all multivariate expressions')

    inc = is_increasing(f, interval, symbol)
    dec = is_decreasing(f, interval, symbol)

    return fuzzy_or([inc, dec])
