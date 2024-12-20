.. _theory:

Theory
======

Let us take a tour on selected key features of ``stalk``, on a symbolic rather than
numerical level. The following presentation is not comprehensive but it hopefully helps to
get a grasp of the functionality and setup.

Parametric subspace
-------------------

Consider a selection of scalar *parameters*: a parameter vector :math:`\mathbf{p}`. The
parameter vector spans a parametric subspace :math:`\mathcal{P}` that maps to a scalar
cost function :math:`X:\mathcal{P} \rightarrow \mathbb{R}`. The optimization problem is then 
stated as follows: Find the the parameter vector that minimizes :math:`X`, namely
:math:`\mathbf{p}^* = \mathrm{argmin}[X(\mathbf{p})]`. Simple as that.

Who gets to choose :math:`\mathcal{P}` and :math:`X`? Us, of course. It's our
optimization problem, so we get to phrase it as we like.

For example, let us consider the water molecule, H\ :sub:`2`\ O. We may choose that our
parameters comprise the bond length between O and H atoms, and the angle between the two O-H
bonds. That is :math:`\mathbf{p}_{H_2O} = [r_{OH}, \alpha]^T`. Further, we may choose that
our cost function :math:`X_{H_2O}` is the chemical binding energy of the composition. This way, the
optimized result :math:`\mathbf{p}_{H_2O}^*` equals to the ground state structural
parameters in the Born-Oppenheimer approximation, based on the theory that was used to
evaluate :math:`X_{H_2O}`.

Alternative, we could phrase the geometry in terms of the O-H bonds and the H-H distance,
because it spans the same parametric subspace with relation to the redundant space - it has
the same number of independent parameters.

Or, we could decide to keep the O-H bond length fixed and only optimize the H-O-H angle.
That would span a different, more constrained parametric subspace. It is generally
okay to reduce the span of the parametric subspace, at the penalty of leaving some
degrees of freedom not optimized that could be meaningful.

What degrees of freedom *are* meaningful?

What if decided to use instead the 9 Cartesian (xyz) coordinates of the atoms? Then, our
optimization problem would be numerically underdetermined and the solution
:math:`\mathbf{p}^*` would not be unique. This is because the binding energy :math:`X` is
(supposedly) invariant to rotation and translation of the system. As a result, an infinite
number of combinations yields the same result and numerical optimization is  bound to fail.

Therefore, our parametric subspace :math:`\{\mathbf{p}\}` needs to be *irreducible*.
Figuring out a set irreducible parameters is only relevant, if the parametric subspace
:math:`{\mathbf{p}}` is connected to another, redundant parameter space, such as full atomic
coordinates. But it can be also tricky, and there are currently no general-purpose tools
provided to perform this. 

It is relatively easy to write a forward mapping, which *reduces* a redundant parameter
space to the irreducible representation. This amounts to *choosing* the parametric subspace
:math:`\{\mathbf{p}\}`, as outlined above.

It can be a bit harder to write a consistent backward mapping, which maps
:math:`\{\mathbf{p}\}` back into the redundant space. Usually, it requires fixing some of the
positions and orientations until the rest are inferred from :math:`\{\mathbf{p}\}`.

For example, it is easy to measure O-H bond length and H-O-H angle from an array that
represents H\ :sub:`2`\ O positions. Writing such as an array based on the bond length and the
angle is possible if, for example, O is fixed at the origin and the H atoms are fixed on the
xy-plane, symmetric about y.

It is vital to keep the mapping *consistent*, meaning that mapping any given set of
parameters back and forth returns the same set of parameters. 

Parameter Hessian and conjugate directions
------------------------------------------

Above, we outlined how to choose the parametric subspace :math:`\{\mathbf{p}\}` and cost
function :math:`X`, resulting in a meaningful optimization problem to find
:math:`\mathbf{p}^* = \mathrm{argmin}[X(\mathbf{p})]`. Then, the optimized cost is 
:math:`X^*=X(\mathbf{p}^*)`. 
We may now expand the cost function in the neighborhood of :math:`X^*` up
the Hessian order as

.. math::
  X \approx X^* + 0.5 (\mathbf{p} - \mathbf{p}^*)^T H (\mathbf{p} - \mathbf{p}^*)

where the parameter Hessian matrix can be expressed as

.. math::
  H = U^T \Lambda U

where the columns of :math:`U` the eigenvectors and :math:`\Lambda` is a diagonal matrix of
the eigenvalues. Thus, the cost landscape can be described in terms of *conjugate
parameters* :math:`\mathbf{d} = U(\mathbf{p} - \mathbf{p}^*)` in *conjugate directions* that
are the columns of :math:`U`. Each conjugate parameters :math:`\mathbf{d}_n` is
characterized by *stiffness* :math:`\lambda_n`, and so the energy landscape can be rephrased
as

.. math::
  X \approx X^* + 0.5 \sum_n \lambda_n d_n^2

Within this neighborhood, it is possible to find :math:`X^*` by doing line minimization
along each :math:`d_n` independently, no more no less. It is highly desired and optimal.
This is why we want to know the Hessian and the conjugate directions.

How do we get to know the Hessian, provided that we already know :math:`X^*`? We can use
a finite-difference method to characterize :math:`H`, such as

.. math::
  H_{nm} \approx \frac{X(\mathbf{p}^* + \sigma_n \mathbf{p}_n + \sigma_m \mathbf{p}_m) - X(\mathbf{p^*})}{\sigma_n \sigma_m}

where :math:`\sigma_n` is a finite displacements and :math:`\mathbf{p}_n` is a unit
displacement along parameter :math:`n`. Then, we calculate the eigenvectors and eigenvalues
of :math:`H`.

Alternatively, one could do a phonon calculation in the redundant space and map the result
to the parameter space.

Line-search
-----------


Parallel line-search
--------------------


Line-search iteration
---------------------


Line-search optimization
------------------------


Parallel line-search optimization
---------------------------------