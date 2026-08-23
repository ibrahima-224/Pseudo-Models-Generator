#!/usr/bin/env python3
"""
Script pour générer les fixtures Safetensors minimales valides pour les tests.

Ce script crée plusieurs fichiers Safetensors avec des configurations différentes
pour tester les modules compare.rs et espec.rs.
"""

import json
import struct
import os
from typing import List, Dict, Any

def create_safetensors_file(path: str, tensors: List[Dict[str, Any]]) -> None:
    """
    Crée un fichier Safetensors minimal valide.
    
    Args:
        path: Chemin du fichier à créer
        tensors: Liste des tenseurs avec name, dtype, shape, data_len
    """
    # Construction de l'en-tête JSON
    header = {}
    offset = 0
    
    for tensor in tensors:
        end_offset = offset + tensor['data_len']
        header[tensor['name']] = {
            'dtype': tensor['dtype'],
            'shape': tensor['shape'],
            'data_offsets': [offset, end_offset]
        }
        offset = end_offset
    
    # Sérialisation JSON
    header_json = json.dumps(header)
    
    # Alignement sur 8 octets
    padding = (8 - (len(header_json) % 8)) % 8
    padded_json = header_json + ' ' * padding
    header_size = len(padded_json)
    
    # Construction du fichier
    file_data = bytearray()
    
    # Ajout de la taille de l'en-tête (8 octets little-endian)
    file_data.extend(struct.pack('<Q', header_size))
    
    # Ajout de l'en-tête JSON
    file_data.extend(padded_json.encode('utf-8'))
    
    # Ajout des données binaires (payload) pour chaque tenseur
    for tensor in tensors:
        # Données simulées (zeros)
        file_data.extend(bytes(tensor['data_len']))
    
    # Écriture du fichier
    with open(path, 'wb') as f:
        f.write(file_data)

def main():
    # Répertoire des fixtures
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures')
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Modèle A : tenseur "weight" de forme [2, 3], F32
    model_a_tensors = [{
        'name': 'weight',
        'dtype': 'F32',
        'shape': [2, 3],
        'data_len': 24  # 2 * 3 * 4 octets
    }]
    create_safetensors_file(os.path.join(fixtures_dir, 'model_a.safetensors'), model_a_tensors)
    
    # Modèle B : identique à A
    model_b_tensors = [{
        'name': 'weight',
        'dtype': 'F32',
        'shape': [2, 3],
        'data_len': 24
    }]
    create_safetensors_file(os.path.join(fixtures_dir, 'model_b.safetensors'), model_b_tensors)
    
    # Modèle C : tenseur "weight" de forme [3, 4], F32
    model_c_tensors = [{
        'name': 'weight',
        'dtype': 'F32',
        'shape': [3, 4],
        'data_len': 48  # 3 * 4 * 4 octets
    }]
    create_safetensors_file(os.path.join(fixtures_dir, 'model_c.safetensors'), model_c_tensors)
    
    # Modèle D : deux tenseurs
    model_d_tensors = [
        {
            'name': 'weight',
            'dtype': 'F32',
            'shape': [2, 3],
            'data_len': 24
        },
        {
            'name': 'bias',
            'dtype': 'F32',
            'shape': [3],
            'data_len': 12  # 3 * 4 octets
        }
    ]
    create_safetensors_file(os.path.join(fixtures_dir, 'model_d.safetensors'), model_d_tensors)
    
    # Modèle E : tenseur "embedding" de forme [10, 5], F16
    model_e_tensors = [{
        'name': 'embedding',
        'dtype': 'F16',
        'shape': [10, 5],
        'data_len': 100  # 10 * 5 * 2 octets
    }]
    create_safetensors_file(os.path.join(fixtures_dir, 'model_e.safetensors'), model_e_tensors)
    
    # Modèle F : tenseur "bias" de forme [2, 3], F32 (nom différent de A)
    model_f_tensors = [{
        'name': 'bias',
        'dtype': 'F32',
        'shape': [2, 3],
        'data_len': 24
    }]
    create_safetensors_file(os.path.join(fixtures_dir, 'model_f.safetensors'), model_f_tensors)
    
    print(f"Fixtures générées avec succès dans {fixtures_dir}")

if __name__ == '__main__':
    main()