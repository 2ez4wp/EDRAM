#!/usr/bin/env python

from __future__ import division, print_function

import logging

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import fuel
import os
import theano
import theano.tensor as T
import time
import cPickle as pickle
from argparse import ArgumentParser
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.algorithms import StepClipping, CompositeRule, RemoveNotFinite, GradientDescent, Adam
from blocks.initialization import Constant, IsotropicGaussian, Uniform
from blocks.graph import ComputationGraph
from recurrent_bn import MLP, BatchNormalization, batch_normalization, SpatialBatchNormalization, get_batch_normalization_updates
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.training import TrackTheBest
from blocks.main_loop import MainLoop
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Identity, Rectifier, Tanh, Softmax
from blocks.bricks.conv import Convolutional, MaxPooling, ConvolutionalSequence
from blocks.bricks.cost import MisclassificationRate
from checkpoint import PartsOnlyCheckpoint, BestCheckpount, PrintingTo
from plot import Plot
from mnist_cluttered import MNISTCluttered
from model import EDRAM

fuel.config.floatX = theano.config.floatX


# ----------------------------------------------------------------------------


def main(name, epochs, batch_size, learning_rate, window_size, conv_sizes, num_filters, fc_dim, enc_dim, dec_dim, step, num_digits, num_classes,
         oldmodel, live_plotting):
    channels, img_height, img_width = 1, 100, 100

    rnninits = {
        'weights_init': Uniform(width=0.02),
        'biases_init': Constant(0.),
    }

    inits = {
        'weights_init': IsotropicGaussian(0.001),
        'biases_init': Constant(0.),
    }

    rec_inits = {
        'weights_init': IsotropicGaussian(0.001),
        'biases_init': Constant(0.),
    }

    convinits = {
        'weights_init': Uniform(width=.2),
        'biases_init': Constant(0.),
    }

    n_iter = step * num_digits
    filter_size1, filter_size2 = zip(conv_sizes, conv_sizes)[:]
    w_height, w_width = window_size.split(',')
    w_height = int(w_height)
    w_width = int(w_width)

    subdir = time.strftime("%Y-%m-%d") + "-" + name
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    lines = ["\n                Running experiment",
             "          subdirectory: %s" % subdir,
             "         learning rate: %g" % learning_rate,
             "        attention size: %s" % window_size,
             "          n_iterations: %d" % n_iter,
             "     encoder dimension: %d" % enc_dim,
             "     decoder dimension: %d" % dec_dim,
             "            batch size: %d" % batch_size,
             "                epochs: %d" % epochs,
             ]
    for line in lines:
        print(line)
    print()

    rectifier = Rectifier()
    conv1 = Convolutional(filter_size=filter_size2, num_filters=int(num_filters / 2), num_channels=channels, image_size=(w_height, w_width), border_mode='half',
                          name='conv1', **convinits)
    conv1_bn = SpatialBatchNormalization(input_dim=(64, 26, 26), conserve_memory=False, n_iter=n_iter, name='conv1_bn')
    conv2 = Convolutional(filter_size=filter_size2, num_channels=int(num_filters / 2), num_filters=int(num_filters / 2), image_size=(26, 26), name='conv2',
                          **convinits)
    conv2_bn = SpatialBatchNormalization(input_dim=(64, 24, 24), conserve_memory=False, n_iter=n_iter, name='conv2_bn')
    max_pooling = MaxPooling(pooling_size=(2, 2), step=(2, 2))
    conv3 = Convolutional(filter_size=filter_size2, num_filters=num_filters, num_channels=int(num_filters / 2), image_size=(12, 12), border_mode='half',
                          name='conv3', **convinits)
    conv3_bn = SpatialBatchNormalization(input_dim=(128, 12, 12), conserve_memory=False, n_iter=n_iter, name='conv3_bn')
    conv4 = Convolutional(filter_size=filter_size2, num_filters=num_filters, num_channels=num_filters, image_size=(12, 12), name='conv4', **convinits)
    conv4_bn = SpatialBatchNormalization(input_dim=(128, 10, 10), conserve_memory=False, n_iter=n_iter, name='conv4_bn')
    conv_mlp = MLP(activations=[Identity()], dims=[12800, fc_dim], name="MLP_conv", **inits)
    conv_mlp_bn = BatchNormalization(input_dim=fc_dim, conserve_memory=False, n_iter=n_iter, name='conv_mlp_bn')
    loc_mlp = MLP(activations=[Identity()], dims=[6, fc_dim], name="MLP_loc", **inits)
    loc_mlp_bn = BatchNormalization(input_dim=fc_dim, conserve_memory=False, n_iter=n_iter, name='loc_mlp_bn')
    encoder_mlp = MLP([Identity()], [fc_dim, 4 * enc_dim], name="MLP_enc", **rec_inits)
    decoder_mlp = MLP([Identity()], [enc_dim, 4 * dec_dim], name="MLP_dec", **rec_inits)
    encoder_rnn = LSTM(activation=Tanh(), dim=enc_dim, name="RNN_enc", **rnninits)

    conv_init = ConvolutionalSequence(
        [Convolutional(filter_size=filter_size1, num_filters=int(num_filters / 8), name='conv1_init'),
         SpatialBatchNormalization(conserve_memory=False, name='conv1_bn_init'),
         Convolutional(filter_size=filter_size2, num_filters=int(num_filters / 8), name='conv2_init'),
         SpatialBatchNormalization(conserve_memory=False, name='conv2_bn_init'),
         Convolutional(filter_size=filter_size2, num_filters=int(num_filters / 4), name='conv3_init'),
         SpatialBatchNormalization(conserve_memory=False, name='conv3_bn_init'),
         ], image_size=(12, 12), num_channels=channels, name='conv_seq_init', **convinits)

    decoder_rnn = LSTM(activation=Tanh(), dim=dec_dim, name="RNN_dec", **rnninits)
    emit_mlp = MLP(activations=[Tanh()], dims=[dec_dim, 6], name='emit_mlp', weights_init=Constant(0.),
                   biases_init=Constant((1., 0., 0., 0., 1., 0.)))

    classification_mlp1 = MLP(activations=[Identity()], dims=[enc_dim, fc_dim], name='MPL_class1', **inits)
    classification_mlp1_bn = BatchNormalization(input_dim=fc_dim, conserve_memory=False, n_iter=n_iter, name='classification_mlp1_bn')
    classification_mlp2 = MLP(activations=[Identity()], dims=[fc_dim, fc_dim], name='MPL_class2', **inits)
    classification_mlp2_bn = BatchNormalization(input_dim=fc_dim, conserve_memory=False, n_iter=n_iter, name='classification_mlp2_bn')
    classification_mlp3 = MLP(activations=[Softmax()], dims=[fc_dim, num_classes], name='MPL_class3', **inits)

    edram = EDRAM(channels=channels, out_height=w_height, out_width=w_width, n_iter=n_iter, num_classes=num_classes, rectifier=rectifier, conv1=conv1,
                  conv1_bn=conv1_bn, conv2=conv2, conv2_bn=conv2_bn, max_pooling=max_pooling, conv3=conv3, conv3_bn=conv3_bn, conv4=conv4, conv4_bn=conv4_bn,
                  conv_mlp=conv_mlp, conv_mlp_bn=conv_mlp_bn, loc_mlp=loc_mlp, loc_mlp_bn=loc_mlp_bn, conv_init=conv_init, encoder_mlp=encoder_mlp,
                  encoder_rnn=encoder_rnn, decoder_mlp=decoder_mlp, decoder_rnn=decoder_rnn, classification_mlp1=classification_mlp1,
                  classification_mlp1_bn=classification_mlp1_bn, classification_mlp2=classification_mlp2, classification_mlp2_bn=classification_mlp2_bn,
                  classification_mlp3=classification_mlp3, emit_mlp=emit_mlp)
    edram.initialize()

    # ------------------------------------------------------------------------
    x = T.ftensor4('features')
    x_coarse = T.ftensor4('features_coarse')
    y = T.ivector('labels')
    wr = T.fmatrix('locations')

    with batch_normalization(edram):
        bn_p, bn_l, m_c1_bn, s_c1_bn, m_c2_bn, s_c2_bn, m_c3_bn, s_c3_bn, m_c4_bn, s_c4_bn, m_c_bn, s_c_bn, m_l_bn, s_l_bn, m_cl1_bn, s_cl1_bn, m_cl2_bn, s_cl2_bn \
            = edram.calculate_train(x, x_coarse)

    def compute_cost(p, wr, y, l):
        cost_where = T.dot(T.sqr(wr - l), [2, 0.5, 2, 0.5, 2, 2])
        cost_y = T.stack([T.nnet.categorical_crossentropy(T.maximum(p[i, :], 1e-7), y) for i in range(0, n_iter)])

        return cost_where, cost_y

    lambda0 = 10.0

    cost_where, cost_y = compute_cost(bn_p, wr, y, bn_l)
    bn_cost = lambda0 * cost_y + cost_where
    bn_cost = bn_cost.sum(axis=0)
    bn_cost = bn_cost.mean()
    bn_cost.name = 'cost'

    bn_error_rate = MisclassificationRate().apply(y, bn_p[-1])
    bn_error_rate.name = 'error_rate'

    # ------------------------------------------------------------
    bn_cg = ComputationGraph([bn_cost, bn_error_rate])

    # Prepare algorithm
    algorithm = GradientDescent(
        cost=bn_cg.outputs[0],
        on_unused_sources='ignore',
        parameters=bn_cg.parameters,
        step_rule=CompositeRule([
            RemoveNotFinite(),
            StepClipping(10.),
            Adam(learning_rate)
        ])
    )

    pop_updates = get_batch_normalization_updates(bn_cg)
    update_params = [conv1_bn.population_mean, conv1_bn.population_stdev, conv2_bn.population_mean, conv2_bn.population_stdev, conv3_bn.population_mean,
                     conv3_bn.population_stdev, conv4_bn.population_mean, conv4_bn.population_stdev, conv_mlp_bn.population_mean, conv_mlp_bn.population_stdev,
                     loc_mlp_bn.population_mean, loc_mlp_bn.population_stdev, classification_mlp1_bn.population_mean, classification_mlp1_bn.population_stdev,
                     classification_mlp2_bn.population_mean, classification_mlp2_bn.population_stdev]
    update_values = [m_c1_bn, s_c1_bn, m_c2_bn, s_c2_bn, m_c3_bn, s_c3_bn, m_c4_bn, s_c4_bn, m_c_bn, s_c_bn, m_l_bn, s_l_bn, m_cl1_bn, s_cl1_bn, m_cl2_bn, s_cl2_bn]

    pop_updates.extend([(p, m) for p, m in zip(update_params, update_values)])

    decay_rate = 0.05
    extra_updates = [(p, m * decay_rate + p * (1 - decay_rate)) for p, m in pop_updates]
    algorithm.add_updates(extra_updates)

    # ------------------------------------------------------------------------
    # Setup monitors

    p, l = edram.calculate_test(x, x_coarse)
    cost_where, cost_y = compute_cost(p, wr, y, l)
    cost = lambda0 * cost_y + cost_where
    cost = cost.sum(axis=0)
    cost = cost.mean()
    cost.name = 'cost'

    error_rate = MisclassificationRate().apply(y, p[-1])
    error_rate.name = 'error_rate'
    monitors = [cost, error_rate]

    plotting_extensions = []
    # Live plotting...
    if live_plotting:
        plot_channels = [
            ['train_cost', 'test_cost'],
            ['train_error_rate', 'test_error_rate'],
        ]
        plotting_extensions = [
            Plot(subdir, channels=plot_channels, server_url='http://155.69.150.60:80/')
        ]

    # ------------------------------------------------------------

    mnist_cluttered_train = MNISTCluttered(which_sets=['train'], sources=('features', 'locations', 'labels'))
    mnist_cluttered_test = MNISTCluttered(which_sets=['test'], sources=('features', 'locations', 'labels'))

    main_loop = MainLoop(
        model=Model([bn_cost]),
        data_stream=DataStream.default_stream(mnist_cluttered_train, iteration_scheme=ShuffledScheme(mnist_cluttered_train.num_examples, batch_size)),
        algorithm=algorithm,
        extensions=[Timing(),
                    FinishAfter(after_n_epochs=epochs),
                    DataStreamMonitoring(monitors,
                                         DataStream.default_stream(mnist_cluttered_train, iteration_scheme=SequentialScheme(mnist_cluttered_train.num_examples,
                                                                                                                            batch_size)), prefix='train'),
                    DataStreamMonitoring(monitors,
                                         DataStream.default_stream(mnist_cluttered_test,
                                                                   iteration_scheme=SequentialScheme(mnist_cluttered_test.num_examples, batch_size)),
                                         prefix="test"),
                    PartsOnlyCheckpoint("{}/{}".format(subdir, name), before_training=False, after_epoch=True, save_separately=['log', ]),
                    TrackTheBest('test_error_rate', 'best_test_error_rate'),
                    BestCheckpount("{}/{}".format(subdir, name), 'best_test_error_rate', save_separately=['model', ]),
                    Printing(),
                    ProgressBar(),
                    PrintingTo("\n".join(lines), "{}/{}_log.txt".format(subdir, name)), ] + plotting_extensions)
    if oldmodel is not None:
        print("Initializing parameters with old model %s" % oldmodel)
        with open(oldmodel, "rb") as f:
            oldmodel = pickle.load(f)
            main_loop.model.set_parameter_values(oldmodel.get_parameter_values())

            main_loop.model.get_top_bricks()[0].conv1_bn.population_mean.set_value(oldmodel.get_top_bricks()[0].conv1_bn.population_mean.get_value())
            main_loop.model.get_top_bricks()[0].conv1_bn.population_stdev.set_value(oldmodel.get_top_bricks()[0].conv1_bn.population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].conv2_bn.population_mean.set_value(oldmodel.get_top_bricks()[0].conv2_bn.population_mean.get_value())
            main_loop.model.get_top_bricks()[0].conv2_bn.population_stdev.set_value(oldmodel.get_top_bricks()[0].conv2_bn.population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].conv3_bn.population_mean.set_value(oldmodel.get_top_bricks()[0].conv3_bn.population_mean.get_value())
            main_loop.model.get_top_bricks()[0].conv3_bn.population_stdev.set_value(oldmodel.get_top_bricks()[0].conv3_bn.population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].conv4_bn.population_mean.set_value(oldmodel.get_top_bricks()[0].conv4_bn.population_mean.get_value())
            main_loop.model.get_top_bricks()[0].conv4_bn.population_stdev.set_value(oldmodel.get_top_bricks()[0].conv4_bn.population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].loc_mlp_bn.population_mean.set_value(oldmodel.get_top_bricks()[0].loc_mlp_bn.population_mean.get_value())
            main_loop.model.get_top_bricks()[0].loc_mlp_bn.population_stdev.set_value(oldmodel.get_top_bricks()[0].loc_mlp_bn.population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].conv_mlp_bn.population_mean.set_value(oldmodel.get_top_bricks()[0].conv_mlp_bn.population_mean.get_value())
            main_loop.model.get_top_bricks()[0].conv_mlp_bn.population_stdev.set_value(oldmodel.get_top_bricks()[0].conv_mlp_bn.population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].classification_mlp1_bn.population_mean.set_value(
                oldmodel.get_top_bricks()[0].classification_mlp1_bn.population_mean.get_value())
            main_loop.model.get_top_bricks()[0].classification_mlp1_bn.population_stdev.set_value(
                oldmodel.get_top_bricks()[0].classification_mlp1_bn.population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].classification_mlp2_bn.population_mean.set_value(
                oldmodel.get_top_bricks()[0].classification_mlp2_bn.population_mean.get_value())
            main_loop.model.get_top_bricks()[0].classification_mlp2_bn.population_stdev.set_value(
                oldmodel.get_top_bricks()[0].classification_mlp2_bn.population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].conv_init.layers[1].population_mean.set_value(
                oldmodel.get_top_bricks()[0].conv_init.layers[1].population_mean.get_value())
            main_loop.model.get_top_bricks()[0].conv_init.layers[1].population_stdev.set_value(
                oldmodel.get_top_bricks()[0].conv_init.layers[1].population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].conv_init.layers[3].population_mean.set_value(
                oldmodel.get_top_bricks()[0].conv_init.layers[3].population_mean.get_value())
            main_loop.model.get_top_bricks()[0].conv_init.layers[3].population_stdev.set_value(
                oldmodel.get_top_bricks()[0].conv_init.layers[3].population_stdev.get_value())

            main_loop.model.get_top_bricks()[0].conv_init.layers[5].population_mean.set_value(
                oldmodel.get_top_bricks()[0].conv_init.layers[5].population_mean.get_value())
            main_loop.model.get_top_bricks()[0].conv_init.layers[5].population_stdev.set_value(
                oldmodel.get_top_bricks()[0].conv_init.layers[5].population_stdev.get_value())
        del oldmodel
    main_loop.run()

    # -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--live-plotting", "--plot", action="store_true",
                        default=True, help="Activate live-plotting to a bokeh-server")
    parser.add_argument("--name", type=str, dest="name",
                        default='MNIST_cluttered_adam_2fc_new_bn', help="Name for this experiment")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=200, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=128, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1e-4, help="Learning rate")
    parser.add_argument("--window_size", "-a", type=str,
                        default='26,26', help="Window size of attention mechanism (height,width)")
    parser.add_argument("--conv-sizes", type=int, nargs='+', dest="conv_sizes",
                        default=[5, 3], help="List of sizes of convolution filters")
    parser.add_argument("--num-filters", type=int, dest="num_filters",
                        default=128, help="Number of filters in convolution")
    parser.add_argument("--fc-dim", type=int, dest="fc_dim",
                        default=1024, help="Fully connected dimension")
    parser.add_argument("--enc-dim", type=int, dest="enc_dim",
                        default=512, help="Encoder RNN state dimension")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                        default=512, help="Decoder  RNN state dimension")
    parser.add_argument("--step", type=int, dest="step",
                        default=10, help="Step size for digit recognition")
    parser.add_argument("--num-digits", type=int, dest="num_digits",
                        default=1, help="Number of digits in the sequence")
    parser.add_argument("--num-classes", type=int, dest="num_classes",
                        default=10, help="Number of classes for recognition")
    parser.add_argument("--oldmodel", type=str, help="Use a model pkl file created by a previous run as a starting point for all parameters")
    args = parser.parse_args()

    main(**vars(args))
