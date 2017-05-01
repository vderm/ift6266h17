#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file dcgan_doc2vec_embed.py

'''

Vasken Dermardiros
For Prof. Aaron Courville, IFT6266 @ UdeM W2017 Semester

Project: In-fill a 64x64 colour image where it's center is missing. Captions
are used to help the task. Whether they aid the process or not is in the eyes
of the beholder.

Full description: https://ift6266h17.wordpress.com/project-description/

TODO
+ new file: try w-gan https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066

FIXME
+ get bilinear interpolation to work

'''

# Dependencies
import os, sys, glob, time
from datetime import datetime
import cPickle as pkl
import numpy as np
import PIL.Image as Image
import lasagne
import theano
import theano.tensor as T

# Paths (global), change these when changing devices
DATA_LOC = '/home/vasken-desktop-linux/Documents/IFT6266_Project/inpainting/'
FTRAIN   = 'train2014'
FVALID   = 'val2014'
CAPTION  = 'dict_key_imgID_value_caps_train_and_valid.pkl'

class Upscale2DLayer(lasagne.layers.Layer):
    '''Attempt at making a bilinear upscaling layer

    '''
    def __init__(self, incoming, scale=2, **kwargs):
        super(Upscale2DLayer, self).__init__(incoming, **kwargs)
        self.scale = scale
    def get_output_for(self, input, **kwargs):
        return T.nnet.abstract_conv.bilinear_upsampling(input, self.scale)
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.scale*input_shape[2], self.scale*input_shape[3])

def iterate_minibatches(inputs, targets, captions, batchsize, shuffle=False, flip=False):
    '''Batch iterator. Option to shuffle and flip images.

    '''
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        inp_exc, target_exc = inputs[excerpt], targets[excerpt]
        caption_exc = [captions[i] for i in excerpt]
        if flip:
            indicesflip = np.random.choice(batchsize, batchsize/2, replace=False)
            inp_exc[indicesflip] = inp_exc[indicesflip, :, :, ::-1]
            target_exc[indicesflip] = target_exc[indicesflip, :, :, ::-1]
        yield inp_exc, target_exc, caption_exc

def load(batch_idx = None, batch_size = None, valid = False, discard_BnW = True):
    '''Load data from FVALID if *valid* is True, otherwise load from FTRAIN.
       Discard black and white examples by default.
    '''
    # Load captions
    caption_path = os.path.join(DATA_LOC, CAPTION)
    with open(caption_path) as cp:
        caption_dict = pkl.load(cp)

    # Image paths
    fname = FVALID if valid else FTRAIN
    data_path = os.path.join(DATA_LOC, fname)
    print data_path + "/*.jpg"
    imgs = glob.glob(data_path + "/*.jpg")
    if batch_idx != None:
        batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]
    else:
        batch_imgs = imgs

    # Initialize variables
    Xraw, yraw, c = [], [], []

    # Loop through and load examples
    for i, img_path in enumerate(batch_imgs):
    # for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        img_array = np.array(img) # 64 x 64 x 3

        cap_id = os.path.basename(img_path)[:-4]

        # Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        else:
            if not discard_BnW:
                input = np.copy(img_array)
                input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

        # Append
        Xraw.append(input)
        yraw.append(target)
        c.append(caption_dict[cap_id])

    # Scale pixels and convert to float32, reshape
    X = np.asarray(Xraw, dtype=np.float32) / 255.
    y = np.asarray(yraw, dtype=np.float32) / 255.
    X = np.transpose(X, (0,3,1,2)) # n x 3 x 64 x 64
    y = np.transpose(y, (0,3,1,2)) # n x 3 x 32 x 32

    print "Examples loaded:", len(X)

    return X, y, c

def draw(X, ygt, ypd, display = True, show_gt = True, return_array = False, save_filename = None):
    '''Draw results: put prediction in the blank middle square, compare to
       ground truth

       PIL.Image takes in (h x w x ch) structure in uint8 data type with range [0,255]
       X: n x 64 x 64 x 3 [<-- PIL input dims, training dim: n x 3 x 64 x 64]
       y: n x 32 x 32 x 3

    '''

    # Rearrange shapes, data type
    X = np.asarray(255*X.transpose(0,2,3,1), np.uint8)
    ygt = np.asarray(255*ygt.transpose(0,2,3,1), np.uint8)
    ypd = np.asarray(255*ypd.transpose(0,2,3,1), np.uint8)

    collage = []

    for i in range(len(X)):
        img_array = X[i,:]
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))

        # Draw ground truth
        # [Image.fromarray(ygt[i,:]).show() for i in range(n)]
        imggt = np.copy(img_array)
        imggt[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = ygt[i,:]

        # Draw predicted
        # [Image.fromarray(ypred[i,:]).show() for i in range(n)]
        imgpd = np.copy(img_array)
        imgpd[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = ypd[i,:]

        # Stack ground truth (if option is set to True) and predicted
        if show_gt: img = np.vstack((imggt, imgpd))
        else:       img = imgpd
        if i == 0: collage = img
        else:      collage = np.hstack((collage, img))

    # Display images
    if display:
        Image.fromarray(collage).show()

    # Save image
    if save_filename is not None:
        Image.fromarray(collage).save(save_filename)

    # Return array
    if return_array:
        return collage # use np.hstack((collage_previous, collage)) to stack images

def lemmatize(c, filename='processed_captions.txt'):
    '''Lemmatize language into root of the words using NLTK package.

    '''
    from string import punctuation
    from copy import deepcopy
    from nltk.stem import WordNetLemmatizer
    wl = WordNetLemmatizer()
    clemon = deepcopy(c)
    for i in range(len(c)):
        cflat = ' '.join(c[i]).translate(None, punctuation).lower()
        clemon[i] = ' '.join([wl.lemmatize(word, pos='v') for word in cflat.split()])
    # Save as text file
    np.savetxt(filename, clemon, fmt='%s')

def train_doc2vec(total_examples, vecSize=256, filename='processed_captions.txt', saveunder='doc2vec.model'):
    '''Train a doc2vec (based on word2vec) embedding given a text file. Each line
    of the file contains an example (sentence, paragraph or document).

    '''
    import gensim
    from gensim import utils
    from gensim.models import Doc2Vec
    import multiprocessing
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    class LabeledLineSentence(object):
        def __init__(self, filename):
            self.filename = filename
        def __iter__(self):
            for uid, line in enumerate(open(filename)):
                yield gensim.models.doc2vec.LabeledSentence(words=line.split(), tags=['SENT_%s' % uid]) # tokenize here...

    it = LabeledLineSentence(filename)
    model = Doc2Vec(size=vecSize, window=10, min_count=5, workers=4)
    model.build_vocab(it)
    model.train(it, total_examples=total_examples, epochs=40, start_alpha=0.025, end_alpha=0.0025)
    model.save(saveunder)
    return model

def load_doc2vec(model_filename='doc2vec.model'):
    '''Load a pretrained doc2vec model.

    '''
    from gensim.models import Doc2Vec
    return Doc2Vec.load(model_filename)

# Inspired: https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
def build_generator(input_var, caption_var, s):
    '''Build the DCGAN generator. The doc2vec embedding is inserted after the
    border image has been encoded. The size of the encoding and doc2vec vector
    length can be changed within the settings.

    '''
    from lasagne.layers import InputLayer, Conv2DLayer, ReshapeLayer, DenseLayer, ConcatLayer
    from lasagne.nonlinearities import sigmoid, LeakyRectify
    from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        print "Failed to load lasagne.layers.dnn.batch_norm_dnn, using lasagne.layers.batch_norm instead."
        from lasagne.layers import batch_norm
    lrelu = LeakyRectify(0.2)
    print "Building Generator"
    # ENCODER
    # input
    image_input = InputLayer(shape=(s['sBatch'], 3, 64, 64), input_var=input_var) # out: 3 x 64 x 64
    # 4 convolutions
    layer = batch_norm(Conv2DLayer(image_input, 1*s['nFilt'], s['sFilt'], stride=s['strid'], pad=s['pad'], nonlinearity=lrelu)) # out: 64 x 32 x 32
    print layer.output_shape
    layer = batch_norm(Conv2DLayer(layer, 2*s['nFilt'], s['sFilt'], stride=s['strid'], pad=s['pad'], nonlinearity=lrelu)) # out: 128 x 16 x 16
    print layer.output_shape
    layer = batch_norm(Conv2DLayer(layer, 4*s['nFilt'], s['sFilt'], stride=s['strid'], pad=s['pad'], nonlinearity=lrelu)) # out: 256 x 8 x 8
    print layer.output_shape
    layer = batch_norm(Conv2DLayer(layer, 8*s['nFilt'], s['sFilt'], stride=s['strid'], pad=s['pad'], nonlinearity=lrelu)) # out: 512 x 4 x 4
    print layer.output_shape
    # flatten and encode
    layer = batch_norm(Conv2DLayer(layer, s['sEncd'], 4, 1, 0, nonlinearity=sigmoid)) # out: encode_size
    layer = ReshapeLayer(layer, ([0], -1)) # flattened
    print layer.output_shape
    # bring in the caption input and merge with image encoding vector
    caption_input = InputLayer(shape=(s['sBatch'], s['sVector']), input_var=caption_var)
    layer = ConcatLayer([layer, caption_input], axis=1) # out: encode_size + doc2vec_embedding_size
    print layer.output_shape
    # DECODER
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 8*s['nFilt']*s['sFilt']**2, nonlinearity=lrelu)) # out: 8192 (=512*4*4)
    print layer.output_shape
    layer = ReshapeLayer(layer, ([0], 8*s['nFilt'], s['sFilt'], s['sFilt']))
    print layer.output_shape
    # 4 fractional-stride convolutions -> 2 options: (1) bilinear upscale then convolve or (2) use even transpose convolutions
    if s['useBilinear']:
        layer = Upscale2DLayer(layer, 2)
        layer = batch_norm(Conv2DLayer(layer, 4*s['nFilt'], 5, stride=1, pad='same', nonlinearity=lrelu)) # out: 256 x 8 x 8
        print layer.output_shape
        layer = Upscale2DLayer(layer, 2)
        layer = batch_norm(Conv2DLayer(layer, 2*s['nFilt'], 5, stride=1, pad='same', nonlinearity=lrelu)) # out: 128 x 16 x 16
        print layer.output_shape
        layer = Upscale2DLayer(layer, 2)
        # layer = batch_norm(Conv2DLayer(layer, 1*s['nFilt'], 5, stride=1, pad='same', nonlinearity=lrelu)) # out: 64 x 32 x 32 NOTE skip?
        # print layer.output_shape
        # output
        layer = Conv2DLayer(layer, 3, 5, stride=1, pad='same', nonlinearity=sigmoid) # out: 3 x 32 x 32
    else:
        layer = batch_norm(Deconv2DLayer(layer, 4*s['nFilt'], s['sFilt'], stride=s['strid'], crop=s['pad'], nonlinearity=lrelu)) # out: 256 x 8 x 8
        print layer.output_shape
        layer = batch_norm(Deconv2DLayer(layer, 2*s['nFilt'], s['sFilt'], stride=s['strid'], crop=s['pad'], nonlinearity=lrelu)) # out: 128 x 16 x 16
        print layer.output_shape
        if s['addExtraGLayer']:
            layer = batch_norm(Deconv2DLayer(layer, 1*s['nFilt'], s['sFilt'], stride=s['strid'], crop=s['pad'], nonlinearity=lrelu)) # out: 64 x 32 x 32 NOTE skip? (A: 1 of 2)
            print layer.output_shape
            # output
            layer = Conv2DLayer(layer, 3, 5, stride=1, pad='same', nonlinearity=sigmoid) # out: 3 x 32 x 32 NOTE if skipped previous, don't apply this convolution! (A: 2 of 2)
        else:
            # output
            layer = Deconv2DLayer(layer, 3, s['sFilt'], stride=s['strid'], crop=s['pad'], nonlinearity=sigmoid) # out: 3 x 32 x 32 NOTE if used previous, don't apply this convolution! (B: 1 of 1)
    print ("Generator output:", layer.output_shape)
    return layer

def build_discriminator(input_var, caption_var, s):
    '''Build the DCGAN discriminator. The doc2vec embedding is inserted after the
    full image has been encoded. The size of the encoding and doc2vec vector
    length can be changed within the settings.

    '''
    from lasagne.layers import InputLayer, Conv2DLayer, ReshapeLayer, DenseLayer, ConcatLayer
    from lasagne.nonlinearities import sigmoid, LeakyRectify
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        print "Failed to load lasagne.layers.dnn.batch_norm_dnn, using lasagne.layers.batch_norm instead."
        from lasagne.layers import batch_norm
    lrelu = LeakyRectify(0.2)
    print "Building Discriminator"
    # DISCRIMINATOR
    # input
    image_input = InputLayer(shape=(s['sBatch'], 3, 64, 64), input_var=input_var) # out: 3 x 64 x 64
    # 4 convolutions
    layer = batch_norm(Conv2DLayer(image_input, 1*s['nFilt'], s['sFilt'], stride=s['strid'], pad=s['pad'], nonlinearity=lrelu)) # out: 64 x 32 x 32
    print layer.output_shape
    layer = batch_norm(Conv2DLayer(layer, 2*s['nFilt'], s['sFilt'], stride=s['strid'], pad=s['pad'], nonlinearity=lrelu)) # out: 128 x 16 x 16
    print layer.output_shape
    layer = batch_norm(Conv2DLayer(layer, 4*s['nFilt'], s['sFilt'], stride=s['strid'], pad=s['pad'], nonlinearity=lrelu)) # out: 256 x 8 x 8
    print layer.output_shape
    layer = batch_norm(Conv2DLayer(layer, 8*s['nFilt'], s['sFilt'], stride=s['strid'], pad=s['pad'], nonlinearity=lrelu)) # out: 512 x 4 x 4
    print layer.output_shape
    # encode
    layer = batch_norm(Conv2DLayer(layer, s['sEncd'], 4, 1, 0, nonlinearity=sigmoid)) # out: encode_size
    layer = ReshapeLayer(layer, ([0], -1)) # flattened
    print layer.output_shape
    # bring in the caption input and merge with image encoding vector
    caption_input = InputLayer(shape=(s['sBatch'], s['sVector']), input_var=caption_var)
    layer = ConcatLayer([layer, caption_input], axis=1) # out: encode_size + doc2vec_embedding_size
    print layer.output_shape
    # output
    layer = DenseLayer(layer, 1, nonlinearity=sigmoid, b=None) # out: 1
    print ("Discriminator output:", layer.output_shape)
    return layer, image_input, caption_input

def main(settings):
    # Troubleshooting
    if settings['troubleshooting']:
        theano.config.optimizer = 'fast_compile'
        theano.config.exception_verbosity = 'high'

    # Load data
    if not 'X' in vars() or not 'X' in globals():
        if settings['endTrainExample'] == None:
            print "Loading full dataset..."
            X, y, c = load()
        else:
            print "Loading from example %i to %i." %(settings['startTrainExample'], settings['endTrainExample'])
            X, y, c = load(settings['startTrainExample'], settings['endTrainExample'])

    # Doc2Vec vector training or loading
    print "Performing Doc2Vec routines..."
    model_filename = 'doc2vec_c'+str(settings['sVector'])+'.model'
    if settings['trainDoc2Vec'] or not os.path.isfile(model_filename):
        print "Training Doc2Vec model"
        lemmatize(c, filename='processed_captions.txt')
        doc2vec_model = train_doc2vec(len(c), vecSize=settings['sVector'], filename='processed_captions.txt', saveunder=model_filename)
    else:
        print "Loading Doc2Vec model"
        doc2vec_model = load_doc2vec(model_filename=model_filename)
    c_doc2vec = [doc2vec_model.docvecs[i] for i in range(len(c))]
    # estimate doc2vec from validation set
    # lemmatize(cValid, filename='processed_captions_valid.txt')
    # doc2vec_model.infer_vector(cValid_processed[...])

    # Prepare Theano variables
    # noise_var = T.matrix('noise')
    border_var = T.tensor4('border')  # border image, nx64x64x3
    center_var = T.tensor4('center')  # center image, nx32x32x3
    caption_var = T.matrix('caption') # caption transformed via doc2vec, nxsVector
    image_var = T.tensor4('image')    # nx64x64x3, dummy variable
    # full_image_subtensor = T.set_subtensor(border_var[:, :, 16:48, 16:48], center_var)
    # full_image = theano.function([center_var, border_var], full_image_subtensor)

    # Create neural net
    print "Building network..."
    generator = build_generator(border_var, caption_var, settings)
    discriminator, image_input, caption_input = build_discriminator(image_var, caption_var, settings)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator, {image_input: T.set_subtensor(border_var[:, :, 16:48, 16:48], center_var), caption_input: caption_var})
    # Create expression for passing fake data through the discriminator
    generator_out = lasagne.layers.get_output(generator)
    fake_out = lasagne.layers.get_output(discriminator, {image_input: T.set_subtensor(border_var[:, :, 16:48, 16:48], generator_out), caption_input: caption_var})

    # Create loss expression
    generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
    discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 0.9) + # NOTE use 0.9 instead of 1
                          lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()
    contextual_loss = lasagne.objectives.squared_error(generator_out, center_var).mean()
    complete_loss = settings['genLossW']*generator_loss + (1-settings['genLossW'])*contextual_loss

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True) # for real net, the loss here will change
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(settings['learnRate']))
    updates = lasagne.updates.adam(generator_loss, generator_params,
                                   learning_rate=eta, beta1=settings['beta1'])
    updates.update(lasagne.updates.adam(discriminator_loss, discriminator_params,
                                        learning_rate=eta, beta1=settings['beta1']))
    updates.update(lasagne.updates.adam(complete_loss, generator_params,
                                        learning_rate=eta, beta1=settings['beta1']))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    print "Compiling functions..."
    train_fn = theano.function([border_var, center_var, caption_var],
                               [(real_out > .5).mean(),
                                (fake_out < .5).mean(),
                                generator_loss,
                                discriminator_loss,
                                complete_loss], updates=updates)

    # Compile another function generating some data
    gen_fn = theano.function([border_var, caption_var], lasagne.layers.get_output(generator, deterministic=True))

    # Continue training
    # TODO make as an argument
    # print "Loading model..."
    # with np.load('gen_20170428050346_ep60_e2048_c128_extraG.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(generator, param_values)
    # with np.load('dis_20170428050346_ep60_e2048_c128_extraG.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(discriminator, param_values)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    train_err_stats = []
    try:
        for epoch in range(settings['epochs']):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X, y, c_doc2vec, settings['sBatch'], shuffle=True, flip=True):
                inputs, targets, captions = batch
                train_err += np.array(train_fn(inputs, targets, captions))
                train_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, settings['epochs'], time.time() - start_time))
            print("  training loss:\t\t{}".format(train_err / train_batches))
            train_err_stats.append(train_err/train_batches)

            # And finally, we plot some generated data
            if epoch % settings['drawImageEveryNEpochs'] == 0:
                ysample = gen_fn(X[0:settings['sBatch']], c_doc2vec[0:settings['sBatch']])
                n = min(10, settings['sBatch'])
                # If saving training images, build the image array over time, draw the whole image at the end of training
                if settings['saveImagesTrain'] and epoch == 0:
                    img_train = draw(X[0:n,], y[0:n,], ysample[0:n], display=False, show_gt=True, return_array=True)
                elif settings['saveImagesTrain'] and epoch != 0:
                    img_train_temp = draw(X[0:n,], y[0:n,], ysample[0:n], display=False, show_gt=False, return_array=True)
                    img_train = np.vstack((img_train, img_train_temp))
                draw(X[0:n,], y[0:n,], ysample[0:n])

            # After half the epochs, we start decaying the learn rate towards zero
            if epoch >= settings['epochs'] // 2:
                progress = float(epoch) / settings['epochs']
                eta.set_value(lasagne.utils.floatX(settings['learnRate']*2*(1 - progress)))

    # Interrupt training, but don't stop the script!
    except KeyboardInterrupt:
        pass

    # Training complete,
    print "Done training."
    # Create a filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_filename = timestamp + '_ep' + str(settings['epochs']) + '_e' + str(settings['sEncd']) + '_c' + str(settings['sVector'])
    if settings['addExtraGLayer']: save_filename = save_filename + '_extraG'
    if settings['genLossW'] != 0.001: save_filename = save_filename + '_genLossW' + str(settings['genLossW'])
    # Save training error statistics
    train_err_stats = np.hstack((np.arange(settings['epochs']).reshape(-1,1), train_err_stats))
    np.savetxt(save_filename+'.txt', train_err_stats, fmt='%.6e', delimiter=',', header='epochs,real_out,generator_loss,discriminator_loss,complete_loss')
    # Save training images?
    if settings['saveImagesTrain']:
        save_train_img = 'img_train_' + save_filename + '.png'
        Image.fromarray(img_train).save(save_train_img)
    # either save the weights to be used another time, or don't
    if settings['saveWeights']:
        print "Saving weights..."
        sys.setrecursionlimit(10000)
        np.savez('gen_' + save_filename + '.npz', *lasagne.layers.get_all_param_values(generator))
        np.savez('dis_' + save_filename + '.npz', *lasagne.layers.get_all_param_values(discriminator))
        print "Done saving."
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)
    else:
        print "Not saving weights."

    # Draw examples using validation set
    n = min(128, len(X))
    batch_idx = 0
    Xvalid, yvalid, cvalid = load(batch_idx, settings['sBatch'], valid = True)
    lemmatize(cvalid, filename='processed_captions_valid.txt')
    cvalid_lemon = [line.rstrip('\n') for line in open('processed_captions_valid.txt')]
    cvalid_doc2vec = [doc2vec_model.infer_vector(i) for i in cvalid_lemon]
    ypred = gen_fn(Xvalid, cvalid_doc2vec)
    if settings['saveImagesValid']: save_valid_img = 'img_valid_' + save_filename + '.png'
    else:                           save_valid_img = ''
    draw(Xvalid[0:n,], yvalid[0:n,], ypred[0:n,], save_filename=save_valid_img)

    # Interactive session
    if settings['interactive']:
        import code
        code.interact(local = dict(globals(), **locals()))

if __name__ == '__main__':
    import argparse
    def str2bool(v):
        '''Helper function for parsing booleans.'''
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')

    # Parse arguments and override defaults if required
    p = argparse.ArgumentParser()

    # Neural network
    p.add_argument('--nFilt',                 default = 64,     type = int,      help = 'number of filters')
    p.add_argument('--sFilt',                 default = 4,      type = int,      help = 'size')
    p.add_argument('--strid',                 default = 2,      type = int,      help = 'stride')
    p.add_argument('--pad',                   default = 1,      type = int,      help = 'padding')
    p.add_argument('--sEncd',                 default = 2048,   type = int,      help = 'encoding size')
    p.add_argument('--useBilinear',           default = False,  type = str2bool, help = 'use bilinear upscaling in generator decoder section [!failing results!]')
    p.add_argument('--addExtraGLayer',        default = False,  type = str2bool, help = 'add an extra layer at the end of the generator')

    # Training
    p.add_argument('--startTrainExample',     default = 0,      type = int,      help = 'first example in training set')
    p.add_argument('--endTrainExample',       default = None,   type = int,      help = 'last example in training set, if "None", use all data')
    p.add_argument('--saveWeights',           default = True,   type = str2bool, help = 'save the weights?')
    p.add_argument('--epochs',                default = 20,     type = int,      help = 'number of epochs to train on')
    p.add_argument('--sBatch',                default = 128,    type = int,      help = 'size of minibatch')
    p.add_argument('--learnRate',             default = 0.0002, type = float,    help = 'learn rate for ADAM optimizer')
    p.add_argument('--beta1',                 default = 0.5,    type = float,    help = 'beta1 rate for ADAM optimizer')
    p.add_argument('--genLossW',              default = 0.001,  type = float,    help = 'weight to apply to discriminator induced generator loss')

    # doc2vec settings
    p.add_argument('--sVector',               default = 128,    type = int,      help = 'encoding vector length')
    p.add_argument('--trainDoc2Vec',          default = False,  type = str2bool, help = 'train Doc2Vec model? otherwise load previously pre-trained model')

    # Other settings
    p.add_argument('--troubleshooting',       default = False,  type = str2bool, help = 'use theano troubleshooting options')
    p.add_argument('--drawImageEveryNEpochs', default = 1,      type = int,      help = 'draw training set images as training progresses')
    p.add_argument('--saveImagesTrain',       default = True,   type = str2bool,)
    p.add_argument('--saveImagesValid',       default = True,   type = str2bool,)
    p.add_argument('--interactive',           default = False,  type = str2bool, help = 'open interactive shell at the end of the script')

    settings = vars(p.parse_args())
    # print settings

    main(settings)

# ============================================================================
# # Deconv2DLayer class from: https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
# class Deconv2DLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
#             nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
#         super(Deconv2DLayer, self).__init__(incoming, **kwargs)
#         self.num_filters = num_filters
#         self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
#         self.stride = lasagne.utils.as_tuple(stride, 2, int)
#         self.pad = lasagne.utils.as_tuple(pad, 2, int)
#         self.W = self.add_param(lasagne.init.Orthogonal(),
#                 (self.input_shape[1], num_filters) + self.filter_size,
#                 name='W')
#         self.b = self.add_param(lasagne.init.Constant(0),
#                 (num_filters,),
#                 name='b')
#         if nonlinearity is None:
#             nonlinearity = lasagne.nonlinearities.identity
#         self.nonlinearity = nonlinearity
#
#     def get_output_shape_for(self, input_shape):
#         shape = tuple(i*s - 2*p + f - 1
#                 for i, s, p, f in zip(input_shape[2:],
#                                       self.stride,
#                                       self.pad,
#                                       self.filter_size))
#         return (input_shape[0], self.num_filters) + shape
#
#     def get_output_for(self, input, **kwargs):
#         op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
#             imshp=self.output_shape,
#             kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
#             subsample=self.stride, border_mode=self.pad)
#         conved = op(self.W, input, self.output_shape[2:])
#         if self.b is not None:
#             conved += self.b.dimshuffle('x', 0, 'x', 'x')
#         return self.nonlinearity(conved)
