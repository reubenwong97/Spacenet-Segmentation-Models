from .crf_rnn import crfrnn_layer
import tensorflow as tf
import tensorflow.keras as keras

def slap_crf_rnn_layer(model, image_dims=(224, 224), num_classes=2, theta_alpha=160., theta_beta=3.,
                        theta_gamma=3., num_iterations=10, name='crfrnn'):
    input_ = model.input
    x = model.output

    output = crfrnn_layer(image_dims=image_dims,
        num_classes=num_classes,
        theta_alpha=theta_alpha,
        theta_beta=theta_beta,
        theta_gamma=theta_gamma,
        num_iterations=num_iterations,
        name=name)(x)

    model = keras.models.Model(input_, output)

    return model