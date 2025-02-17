#!/usr/bin/env python

from numpy import array, linspace, where
from pytest import raises

from stalk.params.PesFunction import PesFunction
from stalk.util import match_to_tol
from stalk import TargetLineSearch

from ..assets.h2o import get_structure_H2O, get_hessian_H2O, pes_H2O
from ..assets.helper import Gs_N200_M7

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# test TargetLineSearch class
def test_TargetLineSearch_init():

    # Test init with zero input
    tls = TargetLineSearch()
    # TargetLineSearchBase properties
    assert tls.bias_mix == 0.0
    assert tls.bias_order == 1
    assert tls.target_fit.x0 == 0.0
    assert tls.target_fit.y0 == 0.0
    assert tls.target_fit.x0_err == 0.0
    assert tls.target_fit.y0_err == 0.0
    assert not tls.valid_target
    # LineSearch properties
    assert len(tls) == 0
    assert not tls.valid
    assert tls.d is None
    assert tls.direction == 0.0
    assert tls.structure is None
    assert tls.hessian is None
    assert tls.W_max is None
    assert tls.R_max == 0.0
    assert len(tls.grid) == 0
    assert len(tls.offsets) == 0
    assert len(tls.values) == 0
    assert len(tls.errors) == 0
    assert tls.get_shifted_params() is None
    # TargetLineSearch properties
    assert tls.M == 0
    assert tls.N == 0
    assert tls.sigma_opt is None
    assert tls.W_opt is None
    assert tls.R_opt is None
    assert tls.Gs is None
    assert tls.Ws is None
    assert not tls.resampled
    assert not tls.optimized

    # Test init with only offsets input
    structure = get_structure_H2O()
    hessian = get_hessian_H2O()
    W = 0.2
    d = 1
    tls = TargetLineSearch(
        structure=structure,
        hessian=hessian,
        d=d,
        W=W
    )
    assert len(tls) == 7
    # Cannot get adjusted offsets without target fit
    with raises(AssertionError):
        tls.figure_out_adjusted_offsets()
    # end with
    # Bias without valid target cannot be computed
    with raises(AssertionError):
        tls.compute_bias_of(R=1.0)
    # end with
    with raises(AssertionError):
        tls.generate_error_surface()
    # end with
    with raises(AssertionError):
        tls.optimize(0.1)
    # end with
    with raises(AssertionError):
        tls.maximize_sigma(0.1)
    # end with
    with raises(AssertionError):
        tls.statistical_cost()
    # end with

# end def


# test TargetLineSearch class
def test_TargetLineSearch_generate():

    # Test optimization with H2O data
    structure = get_structure_H2O()
    hessian = get_hessian_H2O()
    W = 0.2
    d = 1
    M = 11
    tls = TargetLineSearch(
        structure=structure,
        hessian=hessian,
        d=d,
        W=W,
        M=M,
        pes=PesFunction(pes_H2O),
        interpolate_kind='pchip'
    )
    assert tls.valid
    assert tls.valid_target
    assert not tls.resampled
    assert not tls.optimized
    assert len(tls.grid) == M
    assert len(tls.offsets) == M
    assert len(tls.values) == M
    assert len(tls.errors) == M
    assert tls.W_max == W
    # Try disabling a middle value
    tls.disable_value(M - 3)
    assert tls.valid_W_max == W  # Does not change
    tls.enable_value(M - 3)
    # Try disabling the last value
    tls.disable_value(M - 1)
    assert tls.valid_W_max < W  # is reduced
    # Keeping the value disabled to differentiate valid grid

    # Test compute_bias_of  (bias_mix=0.2)
    bias_ref_mix02 = [
        0.10018722, 0.10030823, 0.10059785, 0.10034146, 0.10020098, 0.10083277,
        0.10113375, 0.10182524, 0.1028858, 0.10394551
    ]
    bias_ref_order3 = [
        0.10018722, 0.10030823, 0.10059785, 0.10034146, 0.10020098, 0.10083277,
        0.10113375, 0.10182524, 0.1028858, 0.10394551
    ]
    Ws_ref = linspace(0.0, tls.valid_W_max, 10)
    Rs_ref = [tls._W_to_R(W) for W in Ws_ref]
    # default
    Ws0, Rs0, bias0 = tls.compute_bias_of(bias_mix=0.2)
    match_to_tol(Ws0, Ws_ref)
    match_to_tol(Rs0, Rs_ref)
    match_to_tol(bias0, bias_ref_mix02)
    # scalar R
    Ws1, Rs1, bias1 = tls.compute_bias_of(bias_mix=0.2, R=Rs_ref[5])
    match_to_tol(Ws1, [Ws_ref[5]])
    match_to_tol(Rs1, [Rs_ref[5]])
    match_to_tol(bias1, [bias_ref_mix02[5]])
    # Array R
    Ws2, Rs2, bias2 = tls.compute_bias_of(bias_mix=0.2, R=Rs_ref)
    match_to_tol(Ws2, Ws_ref)
    match_to_tol(Rs2, Rs_ref)
    match_to_tol(bias2, bias_ref_mix02)
    # scalar W
    Ws3, Rs3, bias3 = tls.compute_bias_of(bias_order=3, W=Ws_ref[3])
    match_to_tol(Ws3, [Ws_ref[3]])
    match_to_tol(Rs3, [Rs_ref[3]])
    match_to_tol(bias3, [bias_ref_order3[3]])
    # Array W
    Ws4, Rs4, bias4 = tls.compute_bias_of(bias_order=3, W=Ws_ref)
    match_to_tol(Ws4, Ws_ref)
    match_to_tol(Rs4, Rs_ref)
    match_to_tol(bias4, bias_ref_order3)

    # Test figure_out_adjusted_offsets
    x_offset = 0.1
    tls.target_fit.x0 = x_offset
    match_to_tol(
        tls.figure_out_adjusted_offsets(R=0.2),
        tls.figure_out_offsets(R=0.2) + x_offset
    )

    # Test generate error surface
    assert not tls.resampled
    assert tls.E_mat is None
    assert tls.W_mat is None
    assert tls.S_mat is None
    assert tls.T_mat is None
    with raises(ValueError):
        # Must provide N > 1
        tls.generate_error_surface()
    # end with
    with raises(ValueError):
        # Must provide N > 1
        tls.generate_error_surface(N=1)
    # end with
    with raises(ValueError):
        # Must provide M > 1
        tls.generate_error_surface(M=2, N=20)
    # end with
    with raises(ValueError):
        # Must provide W_max > 0
        tls.generate_error_surface(M=2, N=20, W_max=0.0)
    # end with
    with raises(ValueError):
        # Must provide sigma_max > 0
        tls.generate_error_surface(M=2, N=20, sigma_max=0.0)
    # end with
    with raises(ValueError):
        # Must provide sigma_max > 0
        tls.generate_error_surface(M=2, N=20, noise_frac=0.0)
    # end with
    # Test default values (noise_frac=0.05)
    N = 20
    M = 5
    tls.target_fit.x0 = 0.0
    tls.generate_error_surface(M=M, N=N)
    W_mat_ref = array([[0., 0.1, 0.2],
                       [0., 0.1, 0.2],
                       [0., 0.1, 0.2]])
    S_mat_ref = array([[0., 0., 0.],
                       [0.005, 0.005, 0.005],
                       [0.010, 0.010, 0.010]])
    # Note: this is not independently controlled. We can only consistently compare the
    # First row of E_mat
    E0_mat_ref = array([[1.29622236e-06, 3.53685589e-04, 1.86921621e-03]])
    match_to_tol(tls.S_mat.max() / tls.W_mat.max(), 0.05)
    match_to_tol(tls.W_mat, W_mat_ref)
    match_to_tol(tls.S_mat, S_mat_ref)
    match_to_tol(tls.E_mat[0], E0_mat_ref[0])
    assert all((tls.T_mat == (W_mat_ref >= S_mat_ref)).flatten())
    # Test non-default values
    W_num = 4
    sigma_num = 5
    W_max = 0.75 * tls.W_max
    noise_frac = 0.1
    # Same M and N result in that Gs are not regenerated
    Gs_old = tls.Gs
    tls.generate_error_surface(
        M=M,
        N=N,
        W_max=W_max,
        W_num=W_num,
        sigma_num=sigma_num,
        noise_frac=noise_frac,
    )
    assert Gs_old is tls.Gs
    W_mat_ref1 = array([[0., 0.05, 0.1, 0.15],
                        [0., 0.05, 0.1, 0.15],
                        [0., 0.05, 0.1, 0.15],
                        [0., 0.05, 0.1, 0.15],
                        [0., 0.05, 0.1, 0.15]])
    S_mat_ref1 = array([[0.00000, 0.00000, 0.00000, 0.00000],
                        [0.00375, 0.00375, 0.00375, 0.00375],
                        [0.00750, 0.00750, 0.00750, 0.00750],
                        [0.01125, 0.01125, 0.01125, 0.01125],
                        [0.01500, 0.01500, 0.01500, 0.01500]])
    E0_mat_ref1 = array([[1.29622236e-06, 1.41974295e-04, 3.53685589e-04, 1.07211477e-03]])
    assert tls.W_mat.max() == W_max
    match_to_tol(tls.S_mat.max() / tls.W_mat.max(), noise_frac)
    match_to_tol(tls.W_mat, W_mat_ref1)
    match_to_tol(tls.S_mat, S_mat_ref1)
    match_to_tol(tls.E_mat[0], E0_mat_ref1[0])
    assert all((tls.T_mat == (W_mat_ref1 >= S_mat_ref1)).flatten())

    # test maximize sigma, errors
    with raises(ValueError):
        tls.maximize_sigma(0.0)
    # end with
    with raises(ValueError):
        tls.maximize_sigma(1e-5, max_rounds=0)
    # end with
    with raises(ValueError):
        tls.maximize_sigma(1e-5, S_resolution=0.0)
    # end with
    with raises(ValueError):
        tls.maximize_sigma(1e-5, W_resolution=0.0)
    # end with
    with raises(AssertionError):
        # Epsilon is > 0 but still too small to be found.
        tls.maximize_sigma(1e-10)
    # end with
    # test maximize sigma, default values
    # (presuming generate_error_surface is called like above)
    # Note: the process is stochastic; a deterministic test is done for optimize() method.
    epsilon = 0.02
    W_opt, sigma_opt = tls.maximize_sigma(epsilon)
    assert W_opt < W_max
    assert sigma_opt > 0.0
# end def


# test TargetLineSearch class
def test_TargetLineSearch_optimize():

    # Test optimize method (start over with predefined Gs)
    tls = TargetLineSearch(
        structure=get_structure_H2O(),
        hessian=get_hessian_H2O(),
        d=0,
        W=0.4,
        M=21,
        pes=PesFunction(pes_H2O),
        interpolate_kind='cubic'
    )
    # Optimization fails for epsilon near zero but exception is captured
    epsilon0 = 1e-10
    tls.optimize(epsilon0, Gs=Gs_N200_M7, fit_kind='pf2')
    assert tls.resampled
    assert not tls.optimized
    # Only one round of generation is done.
    assert tls.E_mat.shape == (3, 3)

    # Test defaults with a reasonable epsilon value
    epsilon1 = 0.03
    W_opt_ref = 0.125
    sigma_opt_ref = 0.0175
    tls.optimize(
        epsilon1,
        Gs=Gs_N200_M7,
    )
    assert tls.resampled
    assert tls.optimized
    assert tls.N == Gs_N200_M7.shape[0]
    assert tls.M == Gs_N200_M7.shape[1]
    assert match_to_tol(tls.W_opt, W_opt_ref)
    assert match_to_tol(tls.sigma_opt, sigma_opt_ref)
    # Semantic quality checks
    xi = where(tls.W_mat[0] == tls.W_opt)[0]
    yi = where(tls.S_mat[:, 0] == tls.sigma_opt)[0]
    assert tls.E_mat[xi, yi] < epsilon1
    assert tls.E_mat[xi, yi + 1] > epsilon1

    # Test with precise parameters and compare against hard-coded reference values
    epsilon2 = 0.02
    tls.target_fit.x0 = 0.0
    tls.target_fit.y0 = -0.5
    tls.optimize(
        epsilon2,
        fit_kind='pf4',
        fraction=0.05,
        Gs=Gs_N200_M7,
        W_num=4,
        sigma_num=4,
        sigma_max=0.1,
        W_resolution=0.04,
        S_resolution=0.03,
        bias_order=2,
        bias_mix=0.1,
        max_rounds=5
    )
    W_mat_ref = array([[0.        , 0.03333333, 0.05      , 0.06666667, 0.08333333,
        0.1, 0.13333333, 0.26666667, 0.4       ],
       [0.0, 0.03333333, 0.05      , 0.06666667, 0.08333333,
        0.1, 0.13333333, 0.26666667, 0.4       ],
       [0.0, 0.03333333, 0.05      , 0.06666667, 0.08333333,
        0.1, 0.13333333, 0.26666667, 0.4       ],
       [0.0, 0.03333333, 0.05      , 0.06666667, 0.08333333,
        0.1, 0.13333333, 0.26666667, 0.4       ]])
    S_mat_ref = array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.03333333, 0.03333333, 0.03333333, 0.03333333, 0.03333333,
        0.03333333, 0.03333333, 0.03333333, 0.03333333],
       [0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.06666667,
        0.06666667, 0.06666667, 0.06666667, 0.06666667],
       [0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
        0.1       , 0.1       , 0.1       , 0.1       ]])
    E_mat_ref = array([[1.15732523e-04, 1.08964926e-03, 2.40513598e-03, 4.27062644e-03,
        6.73653268e-03, 9.87118577e-03, 1.81843155e-02, 8.57680267e-02,
        1.46585845e-01],
       [2.53993295e-04, 1.46050077e-01, 1.56157048e-01, 1.47888969e-01,
        1.07558509e-01, 8.85588428e-02, 7.72718877e-02, 1.16554055e-01,
        1.62465079e-01],
       [2.53993310e-04, 1.59311036e-01, 1.81491642e-01, 2.30317227e-01,
        2.55029304e-01, 2.66249403e-01, 2.74661501e-01, 1.47880012e-01,
        1.80823132e-01],
       [2.53993315e-04, 1.92081147e-01, 1.95978597e-01, 2.42091218e-01,
        2.72863369e-01, 2.98739059e-01, 3.45776645e-01, 2.11029429e-01,
        1.98314576e-01]])
    assert match_to_tol(tls.W_mat, W_mat_ref)
    assert match_to_tol(tls.S_mat, S_mat_ref)
    assert match_to_tol(tls.E_mat, E_mat_ref)

# end def
