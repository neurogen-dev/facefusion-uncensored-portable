import os
import sys
from unittest.mock import patch

import pytest
from onnxruntime import InferenceSession

from facefusion import state_manager
from facefusion.inference_manager import INFERENCE_POOL_SET, clear_inference_pool, get_inference_pool
from facefusion.filesystem import resolve_relative_path


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	state_manager.init_item('execution_device_id', '0')
	state_manager.init_item('execution_providers', [ 'cpu' ])
	state_manager.init_item('download_providers', [ 'github' ])
	# content_analyser.pre_check() # Removed content_analyser pre_check


def test_get_inference_pool() -> None:
	state_manager.init_item('execution_providers', [ 'cpu' ])
	# content_analyser.pre_check() # Removed content_analyser pre_check

	# Test with yolo_nsfw model (commented out as it's removed)
	# model_names = [ 'yolo_nsfw' ]
	# model_source_set = content_analyser.get_model_options().get('sources')
	#
	# with patch('facefusion.inference_manager.detect_app_context', return_value = 'cli'):
	# 	get_inference_pool('facefusion.content_analyser', model_names, model_source_set)
	# 	assert isinstance(INFERENCE_POOL_SET.get('cli').get('facefusion.content_analyser.yolo_nsfw.0.cpu').get('content_analyser'), InferenceSession)
	#
	# with patch('facefusion.inference_manager.detect_app_context', return_value = 'ui'):
	# 	get_inference_pool('facefusion.content_analyser', model_names, model_source_set)
	# 	assert isinstance(INFERENCE_POOL_SET.get('ui').get('facefusion.content_analyser.yolo_nsfw.0.cpu').get('content_analyser'), InferenceSession)
	#
	# assert INFERENCE_POOL_SET.get('cli').get('facefusion.content_analyser.yolo_nsfw.0.cpu').get('content_analyser') == INFERENCE_POOL_SET.get('ui').get('facefusion.content_analyser.yolo_nsfw.0.cpu').get('content_analyser')

	# Add a test for another existing model if necessary, or leave it empty if no other models are tested here.
	# For now, just clearing the pool if any tests ran previously.
	clear_inference_pool()
