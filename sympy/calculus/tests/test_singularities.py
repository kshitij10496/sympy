from sympy.calculus.singularities import (singularities, function_range,
                                          continuous_in, is_increasing,
                                          is_strictly_increasing, is_decreasing,
                                          is_strictly_decreasing, is_monotonic)
from sympy.sets import Interval, FiniteSet, EmptySet, Union
from sympy import Symbol, exp, log,  oo, S, I, sqrt, sin, tan, pi

from sympy.utilities.pytest import XFAIL

from sympy.abc import x, y
a = Symbol('a', negative=True)
b = Symbol('b', positive=True)


def test_singularities():
    x = Symbol('x')
    y = Symbol('y')

    assert singularities(x**2, x) == S.EmptySet
    assert singularities(x/(x**2 + 3*x + 2), x) == FiniteSet(-2, -1)
    assert singularities(1/(x**2 + 1), x) == FiniteSet(I, -I)
    assert singularities(x/(x**3 + 1), x) == FiniteSet(-1, (1 - sqrt(3)*I)/2,
                                                       (1 + sqrt(3)*I)/2)
    assert singularities(1/(y**2 + 2*I*y + 1), y) == FiniteSet(-I + sqrt(2)*I, -I - sqrt(2)*I)


@XFAIL
def test_singularities_non_rational():
    x = Symbol('x', real=True)

    assert singularities(exp(1/x), x) == (0)
    assert singularities(log((x - 2)**2), x) == (2)


def test_function_range():
    x = Symbol('x')
    assert function_range(sin(x), x, Interval(-pi/2, pi/2)) == Interval(-1, 1)
    assert function_range(sin(x), x, Interval(0, pi)) == Interval(0, 1)
    assert function_range(tan(x), x, Interval(0, pi)) == Interval(-oo, oo)
    assert function_range(tan(x), x, Interval(pi/2, pi)) == Interval(-oo, 0)
    assert function_range((x + 3)/(x - 2), x, Interval(-5, 5)) == Interval(-oo, oo)
    assert function_range(1/(x**2), x, Interval(-1, 1)) == Interval(1, oo)
    assert function_range(exp(x), x, Interval(-1, 1)) == Interval(exp(-1), exp(1))
    assert function_range(log(x) - x, x, S.Reals) == Interval(-oo, -1)
    assert function_range(sqrt(3*x - 1), x, Interval(0, 2)) == Interval(0, sqrt(5))


def test_continuous_in():
    x = Symbol('x')
    assert continuous_in(sin(x), x, Interval(0, 2*pi)) == Interval(0, 2*pi)
    assert continuous_in(tan(x), x, Interval(0, 2*pi)) == \
        Union(Interval(0, pi/2, False, True), Interval(pi/2, 3*pi/2, True, True),
              Interval(3*pi/2, 2*pi, True, False))
    assert continuous_in((x - 1)/((x - 1)**2), x, S.Reals) == \
        Union(Interval(-oo, 1, True, True), Interval(1, oo, True, True))
    assert continuous_in(log(x) + log(4*x - 1), x, S.Reals) == \
        Interval(1/4, oo, True, True)
    assert continuous_in(1/sqrt(x - 3), x, S.Reals) == Interval(3, oo, True, True)


def test_is_increasing():
    assert is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    assert is_increasing(-x**2, Interval(-oo, 0))
    assert is_increasing(-x**2, Interval(0, oo)) is False
    assert is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3)) is False
    assert is_increasing(x**2 + y, Interval(1, oo), x) is True
    assert is_increasing(-x**2*a, Interval(1, oo), x) is True
    assert is_increasing(1) is True


def test_is_strictly_increasing():
    assert is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))
    assert is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))
    assert is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3)) is False
    assert is_strictly_increasing(-x**2, Interval(0, oo)) is False
    assert is_strictly_decreasing(1) is False


def test_is_decreasing():
    assert is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    assert is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    assert is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2)) is False
    assert is_decreasing(-x**2, Interval(-oo, 0)) is False
    assert is_decreasing(-x**2*b, Interval(-oo, 0), x) is False


def test_is_strictly_decreasing():
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2)) is False
    assert is_strictly_decreasing(-x**2, Interval(-oo, 0)) is False
    assert is_strictly_decreasing(1) is False


def test_is_monotonic():
    assert is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))
    assert is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    assert is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)
    assert is_monotonic(-x**2, S.Reals) is False
    assert is_monotonic(x**2 + y + 1, Interval(1, 2), x) is True
