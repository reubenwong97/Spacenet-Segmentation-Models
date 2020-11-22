from .crfasrnn_keras.src.crfrnn_model import CrfRnnLayer
import tensorflow as tf

def slap_crf_layer(original_model, 
        image_dims=(224, 224),
        num_classes=2,
        theta_alpha=3.,
        theta_beta=160.,
        theta_gamma=3.,
        num_iterations=10,
        name='crfrnn'):
    original_model.trainable = False

    crf_layer = CrfRnnLayer(image_dims=image_dims,
                            num_classes=num_classes,
                            theta_alpha=theta_alpha,
                            theta_beta=theta_beta,
                            theta_gamma=theta_gamma,
                            num_iterations=num_iterations,
                            name=name)([original_model.outputs[0], original_model.inputs[0]])

    new_crf_model = tf.keras.Model(inputs = original_model.input, outputs = crf_layer)

    return(new_crf_model)