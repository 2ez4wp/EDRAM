import theano
import theano.tensor as T
from blocks.filter import get_application_call
from blocks.bricks.base import application
from blocks.bricks import Initializable
from blocks.bricks.recurrent import BaseRecurrent, recurrent


class EDRAM(BaseRecurrent, Initializable):
    def __init__(self, channels, out_height, out_width, n_iter, num_classes, rectifier, conv1, conv1_bn, conv2, conv2_bn, max_pooling, conv3, conv3_bn, conv4,
                 conv4_bn, conv_mlp, conv_mlp_bn, conv_init, loc_mlp, loc_mlp_bn, encoder_mlp, encoder_rnn, decoder_mlp, decoder_rnn, classification_mlp1,
                 classification_mlp1_bn, classification_mlp2, classification_mlp2_bn, classification_mlp3, emit_mlp,
                 **kwargs):
        super(EDRAM, self).__init__(**kwargs)

        self.n_iter = n_iter
        self.channels = channels
        self.out_height = out_height
        self.out_width = out_width
        self.num_classes = num_classes
        self.rectifier = rectifier
        self.conv1 = conv1
        self.conv1_bn = conv1_bn
        self.conv2 = conv2
        self.conv2_bn = conv2_bn
        self.max_pooling = max_pooling
        self.conv3 = conv3
        self.conv3_bn = conv3_bn
        self.conv4 = conv4
        self.conv4_bn = conv4_bn
        self.conv_mlp = conv_mlp
        self.conv_mlp_bn = conv_mlp_bn
        self.loc_mlp = loc_mlp
        self.loc_mlp_bn = loc_mlp_bn
        self.encoder_mlp = encoder_mlp
        self.encoder_rnn = encoder_rnn
        self.decoder_mlp = decoder_mlp
        self.decoder_rnn = decoder_rnn
        self.conv_init = conv_init
        self.classification_mlp1 = classification_mlp1
        self.classification_mlp1_bn = classification_mlp1_bn
        self.classification_mlp2 = classification_mlp2
        self.classification_mlp2_bn = classification_mlp2_bn
        self.classification_mlp3 = classification_mlp3
        self.emit_mlp = emit_mlp

        self.children = [self.rectifier, self.conv1, self.conv1_bn, self.conv2, self.conv2_bn, self.max_pooling, self.conv3, self.conv3_bn, self.conv4,
                         self.conv4_bn, self.conv_mlp, self.conv_mlp_bn, self.loc_mlp, self.loc_mlp_bn, self.encoder_mlp, self.encoder_rnn, self.decoder_mlp,
                         self.decoder_rnn, self.conv_init, self.classification_mlp1, self.classification_mlp1_bn, self.classification_mlp2,
                         self.classification_mlp2_bn, self.classification_mlp3, self.emit_mlp]

    def get_dim(self, name):
        if name == 'h_enc':
            return self.encoder_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.encoder_rnn.get_dim('cells')
        elif name == 'h_dec':
            return self.decoder_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.decoder_rnn.get_dim('cells')
        elif name == 'p':
            return self.num_classes
        elif name == 'l' or name == 'l_cost':
            return 6
        elif name == 'r':
            return self.channels * self.out_height * self.out_width
        else:
            super(EDRAM, self).get_dim(name)

    def emmit_location(self, h_dec):
        l = self.emit_mlp.apply(h_dec)
        l = T.stack((T.clip(l[:, 0], .0, 1.0), l[:, 1], l[:, 2], l[:, 3], T.clip(l[:, 4], .0, 1.0), l[:, 5]), axis=1)
        return l

    @recurrent(sequences=[], contexts=['x', 'x_coarse', 'n_steps', 'batch_size'], states=['l_cost', 'l', 'h_enc', 'c_enc', 'h_dec', 'c_dec'],
               outputs=['p', 'l_cost', 'l', 'r', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'm_c1_bn', 's_c1_bn', 'm_c2_bn', 's_c2_bn', 'm_c3_bn', 's_c3_bn',
                        'm_c4_bn', 's_c4_bn', 'm_c_bn', 's_c_bn', 'm_l_bn', 's_l_bn', 'm_cl1_bn', 's_cl1_bn', 'm_cl2_bn', 's_cl2_bn'])
    def apply(self, l_cost, l, h_enc, c_enc, h_dec, c_dec, x, x_coarse, n_steps, batch_size):
        r = self.transform(l, x)

        c1 = self.conv1.apply(r)
        c1_bn = self.conv1_bn.apply(c1)
        c1_activation = self.rectifier.apply(c1_bn)

        c2 = self.conv2.apply(c1_activation)
        c2_bn = self.conv2_bn.apply(c2)
        c2_activation = self.rectifier.apply(c2_bn)

        pool = self.max_pooling.apply(c2_activation)

        c3 = self.conv3.apply(pool)
        c3_bn = self.conv3_bn.apply(c3)
        c3_activation = self.rectifier.apply(c3_bn)

        c4 = self.conv4.apply(c3_activation)
        c4_bn = self.conv4_bn.apply(c4)
        c4_activation = self.rectifier.apply(c4_bn)

        g_image = self.conv_mlp.apply(c4_activation.flatten(ndim=2))
        g_image_bn = self.conv_mlp_bn.apply(g_image)
        g_image_activation = self.rectifier.apply(g_image_bn)

        g_loc = self.loc_mlp.apply(l)
        g_loc_bn = self.loc_mlp_bn.apply(g_loc)
        g_loc_activation = self.rectifier.apply(g_loc_bn)

        g = g_image_activation * g_loc_activation

        i_enc = self.encoder_mlp.apply(g)
        h_enc, c_enc = self.encoder_rnn.apply(states=h_enc, cells=c_enc, inputs=i_enc, iterate=False)

        cl1 = self.classification_mlp1.apply(h_enc)
        cl1_bn = self.classification_mlp1_bn.apply(cl1)
        cl1_activation = self.rectifier.apply(cl1_bn)

        cl2 = self.classification_mlp2.apply(cl1_activation)
        cl2_bn = self.classification_mlp1_bn.apply(cl2)
        cl2_activation = self.rectifier.apply(cl2_bn)

        p = self.classification_mlp3.apply(cl2_activation)

        i_dec = self.decoder_mlp.apply(h_enc)
        h_dec, c_dec = self.decoder_rnn.apply(states=h_dec, cells=c_dec, inputs=i_dec, iterate=False)

        l_next = self.emmit_location(h_dec)

        m_c1_bn, s_c1_bn = self.get_metadata(c1_bn)
        m_c2_bn, s_c2_bn = self.get_metadata(c2_bn)
        m_c3_bn, s_c3_bn = self.get_metadata(c3_bn)
        m_c4_bn, s_c4_bn = self.get_metadata(c4_bn)
        m_c_bn, s_c_bn = self.get_metadata(g_image_bn)
        m_l_bn, s_l_bn = self.get_metadata(g_loc_bn)
        m_cl1_bn, s_cl1_bn = self.get_metadata(cl1_bn)
        m_cl2_bn, s_cl2_bn = self.get_metadata(cl2_bn)

        return p, l, l_next, r, h_enc, c_enc, h_dec, c_dec, m_c1_bn, s_c1_bn, m_c2_bn, s_c2_bn, m_c3_bn, s_c3_bn, m_c4_bn, s_c4_bn, \
               m_c_bn, s_c_bn, m_l_bn, s_l_bn, m_cl1_bn, s_cl1_bn, m_cl2_bn, s_cl2_bn

    def get_metadata(self, var):
        app_call = get_application_call(var)
        return app_call.metadata['offset'], get_application_call(var).metadata['divisor']

    @application(inputs=['features', 'features_coarse'], outputs=['p', 'l', 'm_c1_bn', 's_c1_bn', 'm_c2_bn', 's_c2_bn', 'm_c3_bn', 's_c3_bn',
                                                                  'm_c4_bn', 's_c4_bn', 'm_c_bn', 's_c_bn', 'm_l_bn', 's_l_bn', 'm_cl1_bn',
                                                                  's_cl1_bn', 'm_cl2_bn', 's_cl2_bn'])
    def calculate_train(self, features, features_coarse):
        batch_size = features.shape[0]

        p, l_cost, l, r, h_enc, c_enc, h_dec, c_dec, m_c1_bn, s_c1_bn, m_c2_bn, s_c2_bn, m_c3_bn, s_c3_bn, m_c4_bn, s_c4_bn, \
        m_c_bn, s_c_bn, m_l_bn, s_l_bn, m_cl1_bn, s_cl1_bn, m_cl2_bn, s_cl2_bn = self.apply(x=features, x_coarse=features_coarse, n_steps=self.n_iter,
                                                                                            batch_size=batch_size)

        return p, l_cost, m_c1_bn, s_c1_bn, m_c2_bn, s_c2_bn, m_c3_bn, s_c3_bn, m_c4_bn, s_c4_bn, m_c_bn, s_c_bn, m_l_bn, s_l_bn, m_cl1_bn, s_cl1_bn, m_cl2_bn, \
               s_cl2_bn

    @application(inputs=['features', 'features_coarse'], outputs=['p', 'l'])
    def calculate_test(self, features, features_coarse):
        batch_size = features.shape[0]

        p, l_cost, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self.apply(x=features, x_coarse=features_coarse,
                                                                                                 n_steps=self.n_iter,
                                                                                                 batch_size=batch_size)
        return p, l_cost

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        result = []
        x = kwargs['x_coarse']
        c = self.conv_init.apply(x)
        h_dec_init = c.reshape((batch_size, -1))
        l_init = self.emmit_location(h_dec_init)

        for state in self.apply.states:
            dim = self.get_dim(state)
            if state == 'h_dec':
                result.append(h_dec_init)
            elif state == 'l':
                result.append(l_init)
            else:
                if dim == 0:
                    result.append(T.zeros((batch_size,)))
                else:
                    result.append(T.zeros((batch_size, dim)))
        return result

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.apply.states

    def transform(self, theta, input):
        num_batch, num_channels, height, width = input.shape
        theta = T.reshape(theta, (-1, 2, 3))

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        grid = self.meshgrid(self.out_height, self.out_width)

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = T.dot(theta, grid)
        x_s = T_g[:, 0]
        y_s = T_g[:, 1]
        x_s_flat = x_s.flatten()
        y_s_flat = y_s.flatten()

        # dimshuffle input to  (bs, height, width, channels)
        input_dim = input.dimshuffle(0, 2, 3, 1)
        input_transformed = self.interpolate(input_dim, x_s_flat, y_s_flat,
                                             self.out_height, self.out_width)

        output = T.reshape(input_transformed,
                           (num_batch, self.out_height, self.out_width, num_channels))
        output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
        return output

    def interpolate(self, im, x, y, out_height, out_width):
        # *_f are floats
        num_batch, height, width, channels = im.shape
        height_f = T.cast(height, theano.config.floatX)
        width_f = T.cast(width, theano.config.floatX)

        # clip coordinates to [-1, 1]
        x = T.clip(x, -1, 1)
        y = T.clip(y, -1, 1)

        # scale coordinates from [-1, 1] to [0, width/height - 1]
        x = (x + 1) / 2 * (width_f - 1)
        y = (y + 1) / 2 * (height_f - 1)

        # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
        # we need those in floatX for interpolation and in int64 for indexing. for
        # indexing, we need to take care they do not extend past the image.
        x0_f = T.floor(x)
        y0_f = T.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1
        x0 = T.cast(x0_f, 'int64')
        y0 = T.cast(y0_f, 'int64')
        x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
        y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

        # The input is [num_batch, height, width, channels]. We do the lookup in
        # the flattened input, i.e [num_batch*height*width, channels]. We need
        # to offset all indices to match the flat version
        dim2 = width
        dim1 = width * height
        base = T.repeat(
            T.arange(num_batch, dtype='int64') * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels for all samples
        im_flat = im.reshape((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # calculate interpolated values
        wa = ((x1_f - x) * (y1_f - y)).dimshuffle(0, 'x')
        wb = ((x1_f - x) * (y - y0_f)).dimshuffle(0, 'x')
        wc = ((x - x0_f) * (y1_f - y)).dimshuffle(0, 'x')
        wd = ((x - x0_f) * (y - y0_f)).dimshuffle(0, 'x')
        output = T.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
        return output

    def linspace(self, start, stop, num):
        # Theano linspace. Behaves similar to np.linspace
        start = T.cast(start, theano.config.floatX)
        stop = T.cast(stop, theano.config.floatX)
        num = T.cast(num, theano.config.floatX)
        step = (stop - start) / (num - 1)
        return T.arange(num, dtype=theano.config.floatX) * step + start

    def meshgrid(self, height, width):
        # This function is the grid generator from eq. (1) in reference [1].
        # It is equivalent to the following numpy code:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        # It is implemented in Theano instead to support symbolic grid sizes.
        # Note: If the image size is known at layer construction time, we could
        # compute the meshgrid offline in numpy instead of doing it dynamically
        # in Theano. However, it hardly affected performance when we tried.
        x_t = T.dot(T.ones((height, 1)),
                    self.linspace(-1.0, 1.0, width).dimshuffle('x', 0))
        y_t = T.dot(self.linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                    T.ones((1, width)))

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        ones = T.ones_like(x_t_flat)
        grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
        return grid
