import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_dcnn_model(input_shape=(160, 160, 3)):
    """
    Builds the custom Dense CNN (D-CNN) architecture as specified.
    """
    model = models.Sequential()

    # Input Layer is implicit in the first layer's input_shape

    # --- Block 1 ---
    model.add(layers.Conv2D(8, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())

    # --- Block 2 ---
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # --- Block 3 ---
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # --- Block 4 ---
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # --- Block 5 (Deep Features) ---
    model.add(layers.Conv2D(128, (5, 5), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # --- Block 6 ---
    model.add(layers.Conv2D(256, (5, 5), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # --- Classification Head ---
    model.add(layers.Flatten())
    
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(32))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(16))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(16))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compilation
    optimizer = optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test the model build
    model = build_dcnn_model()
    model.summary()



































