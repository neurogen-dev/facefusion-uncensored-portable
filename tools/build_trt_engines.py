#!/usr/bin/env python3
"""
Скрипт обхода директории models/ и сборки TensorRT-движков
в форматах fp4, fp8 и fp16. Использует функцию onnx_to_trt из Rope.EngineBuilder.
"""
import os
import sys
import logging
from pathlib import Path
import onnx
from subprocess import run, CalledProcessError
import json
from onnx import version_converter, save_model

# Добавляем корневую директорию в PYTHONPATH для импорта rope
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Настройка логирования до использования logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Пытаемся добавить папку с библиотеками TensorRT в PATH для плагинов
cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_PATH_V12_8')
if cuda_path:
    # Возможные пути установки TensorRT
    for sub in ['tensorrt/lib', 'lib', os.path.join('tensorrt', 'lib', 'x64')]:
        trt_lib = os.path.join(cuda_path, sub)
        if os.path.isdir(trt_lib):
            os.environ['PATH'] = trt_lib + os.pathsep + os.environ.get('PATH', '')
            logger.info(f"Добавлен путь к TensorRT плагинам в PATH: {trt_lib}")
            break

# Папка с исходными ONNX-моделями и выходной шаблон
MODEL_DIR = ROOT_DIR / 'models'
OUTPUT_ROOT = ROOT_DIR / 'tensorrt-engines'
PRECISIONS = ['fp4', 'fp8', 'fp16']

# Размер пула workspace в МБ под RTX 4090 (~20GB из 24GB)
WORKSPACE_POOL_SIZE_MB = 20480

# Целевая версия Opset для конвертации моделей
TARGET_OPSET = 21


def find_onnx_files(root_dir: Path):
    """Ищет все .onnx-файлы в указанной папке"""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.onnx'):
                yield Path(dirpath) / fname


def build_engines():
    total = 0
    # Загружаем результаты инспекции моделей
    inspect_json_path = ROOT_DIR / 'model_inspect.json'
    try:
        with open(inspect_json_path, 'r', encoding='utf-8') as jf:
            inspect_data = json.load(jf)
    except Exception:
        inspect_data = {}
        logger.warning(f"Не удалось загрузить {inspect_json_path}")
    for onnx_path in find_onnx_files(MODEL_DIR):
        total += 1
        rel_path = onnx_path.relative_to(MODEL_DIR)
        base_name = rel_path.with_suffix('')
        # Конвертация Opset при необходимости
        try:
            model0 = onnx.load(str(onnx_path))
            current = model0.opset_import[0].version if model0.opset_import else None
            if current and current < TARGET_OPSET:
                try:
                    model0 = version_converter.convert_version(model0, TARGET_OPSET)
                    save_model(model0, str(onnx_path))
                    logger.info(f"Конвертировал {rel_path} opset {current}->{TARGET_OPSET}")
                except Exception as ce:
                    logger.warning(f"Не удалось конвертировать {rel_path}: {ce}")
            # Получаем имя входа и определяем, есть ли динамические оси
            inp = model0.graph.input[0]
            shape_proto = inp.type.tensor_type.shape
            # Заменяем нулевые значения на 1 для формирования спецификации
            dims = [d.dim_value if (d.dim_value > 0) else 1 for d in shape_proto.dim]
            # Есть ли динамические оси (dim_param или dim_value==0)
            has_dynamic = any((d.dim_param and d.dim_param != '') or d.dim_value == 0 for d in shape_proto.dim)
            # Генерируем спецификацию профиля только для динамических моделей
            if has_dynamic:
                shape_spec = f"{inp.name}:{'x'.join(map(str, dims))}"
            else:
                shape_spec = None
        except Exception as e:
            logger.warning(f"Не удалось получить shape из {rel_path}, пропускаем продвинутую оптимизацию: {e}")
            shape_spec = None
        for precision in PRECISIONS:
            out_path = OUTPUT_ROOT / precision / base_name.with_suffix('.trt')
            if out_path.exists():
                logger.info(f"Пропущено (есть): [{precision}] {rel_path}")
                continue
            # Формируем команду trtexec
            cmd = [
                'trtexec',
                f'--onnx={onnx_path}',
                f'--saveEngine={out_path}',
                # Флаг для ограничения рабочего пула памяти (user-defined workspace size)
                f'--memPoolSize=workspace:{WORKSPACE_POOL_SIZE_MB}M',
                '--noTF32',
                '--profilingVerbosity=none'
            ]
            # Добавляем флаги IO Formats на основе инспекции
            entry = inspect_data.get(str(rel_path))
            if entry and isinstance(entry, dict) and entry.get('inputs'):
                io_specs = []
                for name, info in entry['inputs'].items():
                    shape = info.get('shape', [])
                    # Получаем numpy dtype и маппим на TRT тип
                    dtype_name = info.get('dtype', 'float32')
                    dtype_map = {
                        'float32': 'fp32',
                        'float16': 'fp16',
                        'int8': 'int8',
                        'int4': 'int4',
                        'uint8': 'int8'
                    }
                    trt_dtype = dtype_map.get(dtype_name)
                    # Только для 4D входов (NCHW) и поддерживаемых dtype
                    if trt_dtype and len(shape) == 4:
                        io_specs.append(f"{trt_dtype}:chw")
                if io_specs:
                    spec_str = ','.join(io_specs)
                    cmd.append(f'--inputIOFormats={spec_str}')
                    cmd.append(f'--outputIOFormats={spec_str}')
            # Добавляем precision-флаг
            if precision == 'fp16':
                cmd.append('--fp16')
            elif precision == 'fp8':
                cmd.append('--fp8')
            elif precision == 'fp4':
                # используем int4 квантование как замену fp4
                cmd.append('--int4')
            # Динамические формы
            if shape_spec:
                cmd += [f'--minShapes={shape_spec}', f'--optShapes={shape_spec}', f'--maxShapes={shape_spec}']
            logger.info(f"Запуск trtexec для [{precision}] {rel_path}")
            try:
                run(cmd, check=True)
                logger.info(f"Успешно: [{precision}] {rel_path}")
            except CalledProcessError as e:
                logger.error(f"trtexec ошибка для {rel_path} ({precision}): {e}")
    logger.info(f"Обработано моделей: {total}")


if __name__ == '__main__':
    build_engines() 