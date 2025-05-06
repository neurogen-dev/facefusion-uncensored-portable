#!/usr/bin/env python3
"""
Скрипт обхода папки models/ и сбора метаданных ONNX-моделей в JSON:
- имя модели
- версия Opset / IR
- входы (имя, dtype, shape)
- выходы (имя, dtype, shape)
- статистика операторов (counter op_type)
- размер файла (bytes, MB)
"""
import os
import json
from pathlib import Path
from collections import Counter
import onnx
from onnx import mapping
import numpy as _np  # локальный импорт для dtype mapping

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'
OUT_FILE = ROOT / 'model_inspect.json'

inspect_data = {}
for onnx_path in MODELS_DIR.rglob('*.onnx'):
    rel = onnx_path.relative_to(MODELS_DIR)
    try:
        model = onnx.load(str(onnx_path))
        # Opset and IR
        opset = model.opset_import[0].version if model.opset_import else None
        ir_ver = model.ir_version
        # Inputs
        inputs = {}
        for inp in model.graph.input:
            t = inp.type.tensor_type
            dims = [d.dim_value for d in t.shape.dim]
            np_type = mapping.TENSOR_TYPE_TO_NP_TYPE.get(t.elem_type, None)
            if np_type is not None:
                try:
                    dtype = _np.dtype(np_type).name
                except Exception:
                    dtype = str(np_type)
            else:
                dtype = 'unknown'
            inputs[inp.name] = {'shape': dims, 'dtype': dtype}
        # Outputs
        outputs = {}
        for out in model.graph.output:
            t = out.type.tensor_type
            dims = [d.dim_value for d in t.shape.dim]
            np_type = mapping.TENSOR_TYPE_TO_NP_TYPE.get(t.elem_type, None)
            if np_type is not None:
                try:
                    dtype = _np.dtype(np_type).name
                except Exception:
                    dtype = str(np_type)
            else:
                dtype = 'unknown'
            outputs[out.name] = {'shape': dims, 'dtype': dtype}
        # Count ops
        ops = [node.op_type for node in model.graph.node]
        op_counts = dict(Counter(ops))
        # File size
        size_bytes = os.path.getsize(str(onnx_path))
        size_mb = round(size_bytes / (1024**2), 2)
        # Сборка данных
        inspect_data[str(rel)] = {
            'opset_version': opset,
            'ir_version': ir_ver,
            'inputs': inputs,
            'outputs': outputs,
            'op_counts': op_counts,
            'size_bytes': size_bytes,
            'size_mb': size_mb
        }
    except Exception as e:
        inspect_data[str(rel)] = {'error': str(e)}

# Сохраняем JSON
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(inspect_data, f, ensure_ascii=False, indent=2)

print(f"Inspection completed, saved to {OUT_FILE}") 