# resnet-cppn-gan-tensorflow

Improvements made for training [Compositional Pattern Producing Network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) as a Generative Model, using Residual Generative Adversarial Networks and Variational Autoencoder techniques to produce high resolution images.

![Morphing2](https://cdn.rawgit.com/hardmaru/resnet-cppn-gan-tensorflow/master/examples/example_sinusoid.gif)

Run `python train.py` from the command line to train from scratch and experiment with different settings.

`sampler.py` can be used inside IPython to interactively see results from the models being trained.

See my blog post at [blog.otoro.net](http://blog.otoro.net/2016/06/02/generating-large-images-from-latent-vectors-part-two/) for more details.

![Morphing1](https://cdn.rawgit.com/hardmaru/resnet-cppn-gan-tensorflow/master/examples/example_linear.gif)

I tested the implementation on TensorFlow 0.80.

Used images2gif.py written by Almar Klein, Ant1, Marius van Voorden.

# License

BSD - images2gif.py

MIT - everything else
