import tensorflow as tf
import numpy as np
import h5py

def create_simple_model():
    """
    Crea un modelo simple basado en las dimensiones que vemos en el error
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 126)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Ajustar según número de clases
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def try_load_weights():
    """
    Intenta cargar solo los pesos del modelo
    """
    try:
        # Cargar labels primero para determinar el número de clases
        labels = np.load('label_encoder_classes.npy', allow_pickle=True)
        num_classes = len(labels)
        print(f"✅ Labels cargados: {labels}")
        print(f"Número de clases: {num_classes}")
        
        # Crear modelo con el número correcto de clases
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(50, 126)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        print("✅ Modelo simple creado")
        
        # Intentar cargar pesos
        try:
            model.load_weights('hand_gesture_model.h5')
            print("✅ Pesos cargados exitosamente")
            return model, labels
        except Exception as e:
            print(f"❌ Error cargando pesos: {e}")
            
            # Intentar método alternativo
            try:
                with h5py.File('hand_gesture_model.h5', 'r') as f:
                    print("Estructura del archivo H5:")
                    def print_structure(name, obj):
                        print(name)
                    f.visititems(print_structure)
                    
            except Exception as e2:
                print(f"❌ Error explorando H5: {e2}")
                
            return None, labels
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        return None, None

if __name__ == "__main__":
    print("🔧 Intentando cargar modelo de forma alternativa...")
    model, labels = try_load_weights()
    
    if model is not None:
        print("✅ ¡Modelo cargado exitosamente!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Guardar el modelo reparado
        model.save('hand_gesture_model_fixed.h5')
        print("✅ Modelo reparado guardado como 'hand_gesture_model_fixed.h5'")
    else:
        print("❌ No se pudo cargar el modelo")