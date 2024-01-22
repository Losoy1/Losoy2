from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, LayerNormalization, MultiHeadAttention, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import Input

class Transformer:
    @staticmethod
    def build(width, height, depth, classes, embed_dim=16, num_heads=1, ff_dim=32):
        input_shape = (height, width, depth)
        inputs = Input(shape=input_shape)

        # Reshape & Flatten the input for Transformer
        x = Reshape((height * width, depth))(inputs)

        # Linear projection of input to match the embedding dimension
        x = Dense(embed_dim)(x)

        # Transformer Block
        for _ in range(2):  # Number of Transformer layers
            # Multi-Head Attention
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
            attention_output = Dropout(0.1)(attention_output)
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)

            # Feed Forward Network
            ffn_output = Dense(ff_dim, activation="relu")(x)
            ffn_output = Dense(embed_dim)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Flatten and Dense layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Output layer
        outputs = Dense(classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        return model
