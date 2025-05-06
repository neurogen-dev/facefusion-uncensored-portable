import os
import onnx
from onnxsim import simplify
import argparse
from onnx import helper

def print_model_inputs(model):
    """
    Stampa i nomi e le forme degli input del modello ONNX.
    """
    print("Input del modello:")
    for input in model.graph.input:
        shape = [dim.dim_value if dim.HasField('dim_value') else dim.dim_param for dim in input.type.tensor_type.shape.dim]
        print(f"Nome input: {input.name}, Shape: {shape}")

def make_input_dynamic(model, input_name):
    """
    Rende l'input specificato dinamico nel modello ONNX.
    
    Args:
        model: Il modello ONNX caricato.
        input_name: Nome dell'input da rendere dinamico.
    """
    # Verifica se l'input esiste nel modello
    input_found = False
    for input in model.graph.input:
        if input.name == input_name:
            input_found = True
            # Imposta il nome della dimensione dinamica
            dim_param = 'num_values'
            
            # Modifica la dimensione dell'input per renderlo dinamico
            if len(input.type.tensor_type.shape.dim) == 1:
                # Se è un tensore con una singola dimensione (ad esempio, [1]), rendilo dinamico
                input.type.tensor_type.shape.dim[0].dim_param = dim_param  # Imposta la dimensione dinamica
            elif len(input.type.tensor_type.shape.dim) == 0:
                # Se è un tensore scalare (ad esempio, []), lo rendiamo dinamico come tensore di dimensione [1]
                input.type.tensor_type.shape.dim.extend([helper.make_tensor_shape_proto_dim(dim_param)])
            print(f"L'input '{input.name}' è stato modificato per essere dinamico.")
            break

    # Rimuovi eventuali inizializzatori associati all'input specificato
    if input_found:
        # Crea una nuova lista di inizializzatori senza quelli da rimuovere
        new_initializers = [init for init in model.graph.initializer if init.name != input_name]
        # Rimuovi tutti gli inizializzatori e aggiungi quelli nuovi
        del model.graph.initializer[:]
        model.graph.initializer.extend(new_initializers)

        # Rimuovi nodi costanti associati all'input specificato
        nodes_to_remove = [node for node in model.graph.node if node.op_type == 'Constant' and node.output[0] == input_name]
        for node in nodes_to_remove:
            model.graph.node.remove(node)

        print(f"Eventuali nodi costanti associati a '{input_name}' sono stati rimossi.")
    else:
        print(f"L'input '{input_name}' non è stato trovato nel modello.")
    return input_found

def simplify_onnx_model(onnx_model_path, onnx_model_dest_path, dynamic_input=None):
    # Carica il modello ONNX
    model = onnx.load(onnx_model_path)

    # Stampa gli input del modello per il debug
    print_model_inputs(model)

    # Semplifica il modello
    try:
        model_simp, check = simplify(model)
        if not check:
            print(f"Il modello semplificato per {onnx_model_path} non è stato validato correttamente")
            return
        else:
            print(f"Modello semplificato correttamente.")
    except Exception as e:
        print(f"Errore nella semplificazione di {onnx_model_path}: {e}")
        return

    # Se è stato specificato un input dinamico e l'input è presente, rendilo dinamico nel modello semplificato
    if dynamic_input:
        input_exists = make_input_dynamic(model_simp, dynamic_input)
        if not input_exists:
            print(f"Salto la modifica per il modello {onnx_model_path} poiché l'input '{dynamic_input}' non è presente.")
            onnx.save(model_simp, onnx_model_dest_path)
            print(f"Modello semplificato salvato in: {onnx_model_dest_path}")
            return

    # Salva il modello modificato dopo la semplificazione
    try:
        onnx.save(model_simp, onnx_model_dest_path)
        print(f"Modello semplificato e modificato salvato in: {onnx_model_dest_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio del modello per {onnx_model_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semplifica modelli DFM/ONNX e modifica input dinamici.')
    parser.add_argument('source_dir', type=str, help='Percorso alla cartella di origine')
    parser.add_argument('dest_dir', type=str, help='Percorso alla cartella di destinazione')
    parser.add_argument('--extension', type=str, default='.onnx', help='Estensione dei file da elaborare (default: .onnx)')
    parser.add_argument('--dynamic_input', type=str, help="Nome dell'input da rendere dinamico (es. morph_value:0)")

    args = parser.parse_args()
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    extension = args.extension
    dynamic_input = args.dynamic_input

    # Assicurati che l'estensione inizi con un punto
    if not extension.startswith('.'):
        extension = '.' + extension

    # Crea la cartella di destinazione se non esiste
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Itera attraverso tutti i file nella cartella di origine
    for filename in os.listdir(source_dir):
        if filename.endswith(extension):
            source_file_path = os.path.join(source_dir, filename)
            dest_file_path = os.path.join(dest_dir, filename)
            try:
                simplify_onnx_model(source_file_path, dest_file_path, dynamic_input)
                print(f"Semplificato e salvato: {filename}")
            except Exception as e:
                print(f"Errore nella semplificazione di {filename}: {e}")