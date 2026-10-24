
def v1(location, layer_index):
    """
    Details:
    - Number of autoencoder latents: 32768
    - Number of training tokens: ~64M
    - Activation function: ReLU
    - L1 regularization strength: 0.01
    - Layer normed inputs: false
    - NeuronRecord files:
        `az://openaipublic/sparse-autoencoder/gpt2-small/{location}/collated_activations/{layer_index}/{latent_index}.json`
    """
    assert location in ["mlp_post_act", "resid_delta_mlp"]
    assert layer_index in range(12)
    return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}/autoencoders/{layer_index}.pt"


# we may release them in the future
