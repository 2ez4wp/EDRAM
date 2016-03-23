import collections
import contextlib
import numpy
from picklable_itertools.extras import equizip
import theano
from theano import tensor
from theano.tensor.nnet import bn

from blocks.graph import add_annotation
from blocks.initialization import Constant
from blocks.roles import (BATCH_NORM_POPULATION_MEAN,
                          BATCH_NORM_POPULATION_STDEV, BATCH_NORM_OFFSET,
                          BATCH_NORM_DIVISOR, BATCH_NORM_MINIBATCH_ESTIMATE,
                          BATCH_NORM_SHIFT_PARAMETER, BATCH_NORM_SCALE_PARAMETER,
                          add_role)
from blocks.utils import (shared_floatx_zeros, shared_floatx,
                          shared_floatx_nans)
from blocks.bricks.base import lazy, application
from blocks.bricks.sequences import Sequence, Feedforward, MLP
from blocks.bricks.interfaces import RNGMixin
from blocks.utils import find_bricks


def _add_batch_axis(var):
    """Prepend a singleton axis to a TensorVariable and name it."""
    new_var = new_var = tensor.shape_padleft(var)
    new_var.name = 'shape_padleft({})'.format(var.name)
    return new_var


def _add_role_and_annotate(var, role, annotations=()):
    """Add a role and zero or more annotations to a variable."""
    add_role(var, role)
    for annotation in annotations:
        add_annotation(var, annotation)


class BatchNormalization(RNGMixin, Feedforward):
    r"""Normalizes activations, parameterizes a scale and shift.

    Parameters
    ----------
    input_dim : int or tuple
        Shape of a single input example. It is assumed that a batch axis
        will be prepended to this.
    broadcastable : tuple, optional
        Tuple the same length as `input_dim` which specifies which of the
        per-example axes should be averaged over to compute means and
        standard deviations. For example, in order to normalize over all
        spatial locations in a `(batch_index, channels, height, width)`
        image, pass `(False, True, True)`.
    conserve_memory : bool, optional
        Use an implementation that stores less intermediate state and
        therefore uses less memory, at the expense of 5-10% speed. Default
        is `True`.
    epsilon : float, optional
       The stabilizing constant for the minibatch standard deviation
       computation (when the brick is run in training mode).
       Added to the variance inside the square root, as in the
       batch normalization paper.
    scale_init : object, optional
        Initialization object to use for the learned scaling parameter
        ($\\gamma$ in [BN]_). By default, uses constant initialization
        of 1.
    shift_init : object, optional
        Initialization object to use for the learned shift parameter
        ($\\beta$ in [BN]_). By default, uses constant initialization of 0.

    Notes
    -----
    In order for trained models to behave sensibly immediately upon
    upon deserialization, by default, this brick runs in *inference* mode,
    using a population mean and population standard deviation (initialized
    to zeros and ones respectively) to normalize activations. It is
    expected that the user will adapt these during training in some
    fashion, independently of the training objective, e.g. by taking a
    moving average of minibatch-wise statistics.

    In order to *train* with batch normalization, one must obtain a
    training graph by transforming the original inference graph. See
    :func:`~blocks.graph.apply_batch_normalization` for a routine to
    transform graphs, and :func:`~blocks.graph.batch_normalization`
    for a context manager that may enable shorter compile times
    (every instance of :class:`BatchNormalization` is itself a context
    manager, entry into which causes applications to be in minibatch
    "training" mode, however it is usually more convenient to use
    :func:`~blocks.graph.batch_normalization` to enable this behaviour
    for all of your graph's :class:`BatchNormalization` bricks at once).

    Note that training in inference mode should be avoided, as this
    brick introduces scales and shift parameters (tagged with the
    `PARAMETER` role) that, in the absence of batch normalization,
    usually makes things unstable. If you must do this, filter for and
    remove `BATCH_NORM_SHIFT_PARAMETER` and `BATCH_NORM_SCALE_PARAMETER`
    from the list of parameters you are training, and this brick should
    behave as a (somewhat expensive) no-op.

    This Brick accepts `scale_init` and `shift_init` arguments but is
    *not* an instance of :class:`~blocks.bricks.Initializable`, and will
    therefore not receive pushed initialization config from any parent
    brick. In almost all cases, you will probably want to stick with the
    defaults (unit scale and zero offset), but you can explicitly pass one
    or both initializers to override this.

    This has the necessary properties to be inserted into a
    :class:`blocks.bricks.conv.ConvolutionalSequence` as-is, in which case
    the `input_dim` should be omitted at construction, to be inferred from
    the layer below.

    """

    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, broadcastable=None,
                 conserve_memory=True, epsilon=1e-4, scale_init=None,
                 shift_init=None, n_iter=None, **kwargs):
        self.input_dim = input_dim
        self.n_iter = n_iter
        self.broadcastable = broadcastable
        self.conserve_memory = conserve_memory
        self.epsilon = epsilon
        self.scale_init = (Constant(1) if scale_init is None
                           else scale_init)
        self.shift_init = (Constant(0) if shift_init is None
                           else shift_init)
        self._training_mode = []
        super(BatchNormalization, self).__init__(**kwargs)

    @application(inputs=['input_', 'i'], outputs=['output'])
    def apply(self, input_, application_call, i=None):
        if self._training_mode:
            mean, stdev = self._compute_training_statistics(input_)
        else:
            mean, stdev = self._prepare_population_statistics(i)
        # Useful for filtration of calls that were already made in
        # training mode when doing graph transformations.
        # Very important to cast to bool, as self._training_mode is
        # normally a list (to support nested context managers), which would
        # otherwise get passed by reference and be remotely mutated.
        application_call.metadata['training_mode'] = bool(self._training_mode)
        # Useful for retrieving a list of updates for population
        # statistics. Ditch the broadcastable first axis, though, to
        # make it the same dimensions as the population mean and stdev
        # shared variables.
        application_call.metadata['offset'] = mean[0]
        application_call.metadata['divisor'] = stdev[0]
        # Give these quantities roles in the graph.
        _add_role_and_annotate(mean, BATCH_NORM_OFFSET,
                               [self, application_call])
        _add_role_and_annotate(stdev, BATCH_NORM_DIVISOR,
                               [self, application_call])
        scale = _add_batch_axis(self.scale)
        shift = _add_batch_axis(self.shift)
        # Heavy lifting is done by the Theano utility function.
        normalized = bn.batch_normalization(input_, scale, shift, mean, stdev,
                                            mode=('low_mem'
                                                  if self.conserve_memory
                                                  else 'high_mem'))
        return normalized

    def __enter__(self):
        self._training_mode.append(True)

    def __exit__(self, *exc_info):
        self._training_mode.pop()

    def _compute_training_statistics(self, input_):
        if self.n_iter:
            axes = (0,) + tuple((i + 1) for i, b in
                                enumerate(self.population_mean[0].broadcastable)
                                if b)
        else:
            axes = (0,) + tuple((i + 1) for i, b in
                                enumerate(self.population_mean.broadcastable)
                                if b)
        mean = input_.mean(axis=axes, keepdims=True)
        if self.n_iter:
            assert mean.broadcastable[1:] == self.population_mean[0].broadcastable
        else:
            assert mean.broadcastable[1:] == self.population_mean.broadcastable
        stdev = tensor.sqrt(tensor.var(input_, axis=axes, keepdims=True) +
                            numpy.cast[theano.config.floatX](self.epsilon))
        if self.n_iter:
            assert stdev.broadcastable[1:] == self.population_stdev[0].broadcastable
        else:
            assert stdev.broadcastable[1:] == self.population_stdev.broadcastable
        add_role(mean, BATCH_NORM_MINIBATCH_ESTIMATE)
        add_role(stdev, BATCH_NORM_MINIBATCH_ESTIMATE)
        return mean, stdev

    def _prepare_population_statistics(self, i):
        if self.n_iter:
            mean = _add_batch_axis(self.population_mean[i])
            stdev = _add_batch_axis(self.population_stdev[i])
        else:
            mean = _add_batch_axis(self.population_mean)
            stdev = _add_batch_axis(self.population_stdev)
        return mean, stdev

    def _allocate(self):
        input_dim = ((self.input_dim,)
                     if not isinstance(self.input_dim, collections.Sequence)
                     else self.input_dim)
        broadcastable = (tuple(False for _ in input_dim)
                         if self.broadcastable is None else self.broadcastable)
        if len(input_dim) != len(broadcastable):
            raise ValueError("input_dim and broadcastable must be same length")
        var_dim = tuple(1 if broadcast else dim for dim, broadcast in
                        equizip(input_dim, broadcastable))
        broadcastable = broadcastable

        # "gamma", from the Ioffe & Szegedy manuscript.
        self.scale = shared_floatx_nans(var_dim, name='batch_norm_scale',
                                        broadcastable=broadcastable)

        # "beta", from the Ioffe & Szegedy manuscript.
        self.shift = shared_floatx_nans(var_dim, name='batch_norm_shift',
                                        broadcastable=broadcastable)
        add_role(self.scale, BATCH_NORM_SCALE_PARAMETER)
        add_role(self.shift, BATCH_NORM_SHIFT_PARAMETER)
        self.parameters.append(self.scale)
        self.parameters.append(self.shift)

        # These aren't technically parameters, in that they should not be
        # learned using the same cost function as other model parameters.
        self.population_mean = shared_floatx_zeros(((self.n_iter,) if self.n_iter else ()) + var_dim,
                                                   name='population_mean',
                                                   broadcastable=((False,) if self.n_iter else ()) + broadcastable)
        self.population_stdev = shared_floatx(numpy.ones(((self.n_iter,) if self.n_iter else ()) + var_dim),
                                              name='population_stdev',
                                              broadcastable=((False,) if self.n_iter else ()) + broadcastable)
        add_role(self.population_mean, BATCH_NORM_POPULATION_MEAN)
        add_role(self.population_stdev, BATCH_NORM_POPULATION_STDEV)

        # Normally these would get annotated by an AnnotatingList, but they
        # aren't in self.parameters.
        add_annotation(self.population_mean, self)
        add_annotation(self.population_stdev, self)

    def _initialize(self):
        self.shift_init.initialize(self.shift, self.rng)
        self.scale_init.initialize(self.scale, self.rng)

    # Needed for the Feedforward interface.
    @property
    def output_dim(self):
        return self.input_dim

    # The following properties allow for BatchNormalization bricks
    # to be used directly inside of a ConvolutionalSequence.
    @property
    def image_size(self):
        return self.input_dim[-2:]

    @image_size.setter
    def image_size(self, value):
        if not isinstance(self.input_dim, collections.Sequence):
            self.input_dim = (None,) + tuple(value)
        else:
            self.input_dim = (self.input_dim[0],) + tuple(value)

    @property
    def num_channels(self):
        return self.input_dim[0]

    @num_channels.setter
    def num_channels(self, value):
        if not isinstance(self.input_dim, collections.Sequence):
            self.input_dim = (value,) + (None, None)
        else:
            self.input_dim = (value,) + self.input_dim[-2:]

    def get_dim(self, name):
        if name in ('input', 'output'):
            return self.input_dim
        else:
            raise KeyError

    @property
    def num_output_channels(self):
        return self.num_channels


class SpatialBatchNormalization(BatchNormalization):
    """Convenient subclass for batch normalization across spatial inputs.

    Parameters
    ----------
    input_dim : int or tuple
        The input size of a single example. Must be length at least 2.
        It's assumed that the first axis of this tuple is a "channels"
        axis, which should not be summed over, and all remaining
        dimensions are spatial dimensions.

    Notes
    -----
    See :class:`BatchNormalization` for more details (and additional
    keyword arguments).

    """

    def _allocate(self):
        if not isinstance(self.input_dim,
                          collections.Sequence) or len(self.input_dim) < 2:
            raise ValueError('expected input_dim to be length >= 2 '
                             'e.g. (channels, height, width)')
        self.broadcastable = (False,) + ((True,) * (len(self.input_dim) - 1))
        super(SpatialBatchNormalization, self)._allocate()


class BatchNormalizedMLP(MLP):
    """Convenient subclass for building an MLP with batch normalization.

    Parameters
    ----------
    conserve_memory : bool, optional
        See :class:`BatchNormalization`.

    Notes
    -----
    All other parameters are the same as :class:`~blocks.bricks.MLP`. Each
    activation brick is wrapped in a :class:`~blocks.bricks.Sequence`
    containing an appropriate :class:`BatchNormalization` brick and
    the activation that follows it.

    By default, the contained :class:`~blocks.bricks.Linear` bricks will
    not contain any biases, as they could be canceled out by the biases
    in the :class:`BatchNormalization` bricks being added. Pass
    `use_bias` with a value of `True` if you really want this for some
    reason.

    """

    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, *args, **kwargs):
        conserve_memory = kwargs.pop('conserve_memory', True)
        activations = [
            Sequence([
                BatchNormalization(conserve_memory=conserve_memory).apply,
                act.apply
            ], name='batch_norm_activation_{}'.format(i))
            for i, act in enumerate(activations)
            ]
        # Batch normalization bricks incorporate a bias, so there's no
        # need for our Linear bricks to have them.
        kwargs.setdefault('use_bias', False)
        super(BatchNormalizedMLP, self).__init__(activations, dims, *args,
                                                 **kwargs)

    @property
    def conserve_memory(self):
        return self._conserve_memory

    @conserve_memory.setter
    def conserve_memory(self, value):
        self._conserve_memory = value
        for act in self.activations:
            assert isinstance(act.children[0], BatchNormalization)
            act.children[0].conserve_memory = value

    def _push_allocation_config(self):
        super(BatchNormalizedMLP, self)._push_allocation_config()
        # Do the extra allocation pushing for the BatchNormalization
        # bricks. They need as their input dimension the output dimension
        # of each linear transformation.  Exclude the first dimension,
        # which is the input dimension.
        for act, dim in equizip(self.activations, self.dims[1:]):
            assert isinstance(act.children[0], BatchNormalization)
            act.children[0].input_dim = dim


@contextlib.contextmanager
def batch_normalization(*bricks):
    r"""Context manager to run batch normalization in "training mode".

    Parameters
    ----------
    \*bricks
        One or more bricks which will be inspected for descendant
        instances of :class:`~blocks.bricks.BatchNormalization`.

    Notes
    -----
    Graph replacement using :func:`apply_batch_normalization`, while
    elegant, can lead to Theano graphs that are quite large and result
    in very slow compiles. This provides an alternative mechanism for
    building the batch normalized training graph. It can be somewhat
    less convenient as it requires building the graph twice if one
    wishes to monitor the output of the inference graph during training.

    Examples
    --------
    First, we'll create a :class:`~blocks.bricks.BatchNormalizedMLP`.

    >>> import theano
    >>> from blocks.bricks import BatchNormalizedMLP, Tanh
    >>> from blocks.initialization import Constant, IsotropicGaussian
    >>> mlp = BatchNormalizedMLP([Tanh(), Tanh()], [4, 5, 6],
    ...                          weights_init=IsotropicGaussian(0.1),
    ...                          biases_init=Constant(0))
    >>> mlp.initialize()

    Now, we'll construct an output variable as we would normally. This
    is getting normalized by the *population* statistics, which by
    default are initialized to 0 (mean) and 1 (standard deviation),
    respectively.

    >>> x = theano.tensor.matrix()
    >>> y = mlp.apply(x)

    And now, to construct an output with batch normalization enabled,
    i.e. normalizing pre-activations using per-minibatch statistics, we
    simply make a similar call inside of a `with` statement:

    >>> with batch_normalization(mlp):
    ...     y_bn = mlp.apply(x)

    Let's verify that these two graphs behave differently on the
    same data:

    >>> import numpy
    >>> data = numpy.arange(12, dtype=theano.config.floatX).reshape(3, 4)
    >>> inf_y = y.eval({x: data})
    >>> trn_y = y_bn.eval({x: data})
    >>> numpy.allclose(inf_y, trn_y)
    False

    """
    bn = find_bricks(bricks, lambda b: isinstance(b, BatchNormalization))
    # Can't use either nested() (deprecated) nor ExitStack (not available
    # on Python 2.7). Well, that sucks.
    try:
        for brick in bn:
            brick.__enter__()
        yield
    finally:
        for brick in bn[::-1]:
            brick.__exit__()


def _training_mode_application_calls(application_calls):
    """Filter for application calls made in 'training mode'."""
    out = []
    for app_call in application_calls:
        assert isinstance(app_call.application.brick, BatchNormalization)
        assert app_call.application.application == BatchNormalization.apply
        if app_call.metadata.get('training_mode', False):
            out.append(app_call)
    return out


def get_batch_normalization_updates(training_graph, allow_duplicates=False):
    """Extract correspondences for learning BN population statistics.

    Parameters
    ----------
    training_graph : :class:`~blocks.graph.ComputationGraph`
        A graph of expressions wherein "training mode" batch normalization
        is taking place.
    allow_duplicates : bool, optional
        If `True`, allow multiple training-mode application calls from the
        same :class:`~blocks.bricks.BatchNormalization` instance, and
        return pairs corresponding to all of them. It's then the user's
        responsibility to do something sensible to resolve the duplicates.

    Returns
    -------
    update_pairs : list of tuples
        A list of 2-tuples where the first element of each tuple is the
        shared variable containing a "population" mean or standard
        deviation, and the second is a Theano variable for the
        corresponding statistics on a minibatch. Note that multiple
        applications of a single :class:`blocks.bricks.BatchNormalization`
        may appear in the graph, and therefore (if `allow_duplicates` is
        True) a single population variable may map to several different
        minibatch variables, and appear multiple times in this mapping.
        This can happen in recurrent models, siamese networks or other
        models that reuse pathways.

    Notes
    -----
    Used in their raw form, these updates will simply overwrite the
    population statistics with the minibatch statistics at every gradient
    step. You will probably want to transform these pairs into something
    more sensible, such as keeping a moving average of minibatch values,
    or accumulating an average over the entire training set once every few
    epochs.

    """
    from toolz import isdistinct
    from functools import partial
    from blocks.roles import OUTPUT
    from blocks.filter import VariableFilter, get_application_call
    var_filter = VariableFilter(bricks=[BatchNormalization], roles=[OUTPUT])
    all_app_calls = map(get_application_call, var_filter(training_graph))
    train_app_calls = _training_mode_application_calls(all_app_calls)
    if len(train_app_calls) == 0:
        raise ValueError("no training mode BatchNormalization "
                         "applications found in graph")
    bricks = [c.application.brick for c in train_app_calls]

    if not allow_duplicates and not isdistinct(bricks):
        raise ValueError('multiple applications of the same '
                         'BatchNormalization brick; pass allow_duplicates '
                         '= True to override this check')

    def extract_pair(brick_attribute, metadata_key, app_call):
        return (getattr(app_call.application.brick, brick_attribute),
                app_call.metadata[metadata_key])

    mean_pair = partial(extract_pair, 'population_mean', 'offset')
    stdev_pair = partial(extract_pair, 'population_stdev', 'divisor')
    return sum([[mean_pair(a), stdev_pair(a)] for a in train_app_calls], [])
