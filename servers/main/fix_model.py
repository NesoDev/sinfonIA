import h5py
import json
import numpy as np

def fix_model_config():
    """
    Intenta arreglar el modelo H5 cambiando batch_shape a input_shape
    """
    model_path = 'hand_gesture_model.h5'
    backup_path = 'hand_gesture_model_backup.h5'
    
    try:
        # Hacer backup del modelo original
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"✅ Backup creado: {backup_path}")
        
        # Abrir el archivo H5
        with h5py.File(model_path, 'r+') as f:
            # Buscar la configuración del modelo
            if 'model_config' in f.attrs:
                config_bytes = f.attrs['model_config']
                if isinstance(config_bytes, bytes):
                    config_str = config_bytes.decode('utf-8')
                else:
                    config_str = config_bytes
                    
                config = json.loads(config_str)
                print("✅ Configuración del modelo encontrada")
                
                # Función recursiva para reemplazar batch_shape con input_shape
                def fix_config_recursive(obj):
                    if isinstance(obj, dict):
                        if 'batch_shape' in obj:
                            batch_shape = obj.pop('batch_shape')
                            obj['input_shape'] = batch_shape[1:] if batch_shape[0] is None else batch_shape
                            print(f"✅ Reemplazado batch_shape {batch_shape} con input_shape {obj['input_shape']}")
                        
                        for key, value in obj.items():
                            fix_config_recursive(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            fix_config_recursive(item)
                
                # Aplicar el fix
                fix_config_recursive(config)
                
                # Guardar la nueva configuración
                new_config_str = json.dumps(config)
                f.attrs['model_config'] = new_config_str.encode('utf-8')
                print("✅ Configuración actualizada guardada")
                
                return True
                
    except Exception as e:
        print(f"❌ Error: {e}")
        # Restaurar backup si algo salió mal
        try:
            shutil.copy2(backup_path, model_path)
            print("✅ Backup restaurado")
        except:
            pass
        return False

if __name__ == "__main__":
    print("🔧 Intentando reparar el modelo...")
    if fix_model_config():
        print("✅ Modelo reparado. Probando carga...")
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model('hand_gesture_model.h5', compile=False)
            print("✅ ¡Modelo cargado exitosamente!")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
        except Exception as e:
            print(f"❌ El modelo aún no se puede cargar: {e}")
    else:
        print("❌ No se pudo reparar el modelo")