from source import vqvae_models, transformer_models

if __name__ == '__main__':
    x = transformer_models.positional_encoding(10, 5)
    print(x)