from functools import lru_cache
from typing import List

import numpy
from tqdm import tqdm

# from facefusion import inference_manager, state_manager, wording # Commented out unused imports
from facefusion import state_manager, wording # Keep used imports
# from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url # Commented out unused imports
from facefusion.filesystem import resolve_relative_path
# from facefusion.thread_helper import conditional_thread_semaphore # Commented out unused imports
# from facefusion.types import Detection, DownloadScope, Fps, InferencePool, ModelOptions, ModelSet, Score, VisionFrame # Commented out unused types
from facefusion.types import DownloadScope, Fps, ModelOptions, ModelSet, VisionFrame # Keep used types
# from facefusion.vision import detect_video_fps, fit_frame, read_image, read_video_frame # Commented out unused imports
from facefusion.vision import detect_video_fps, read_image, read_video_frame # Keep used imports

STREAM_COUNTER = 0


@lru_cache(maxsize = None)
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	# Commented out NSFW model definition
	# return\\
	# {
	# 	\'yolo_nsfw\':
	# 	{
	# 		\'hashes\':
	# 		{
	# 			\'content_analyser\':
	# 			{
	# 				\'url\': resolve_download_url(\'models-3.2.0\', \'yolo_11m_nsfw.hash\'),
	# 				\'path\': resolve_relative_path(\'../.assets/models/yolo_11m_nsfw.hash\')
	# 			}
	# 		},
	# 		\'sources\':
	# 		{
	# 			\'content_analyser\':
	# 			{
	# 				\'url\': resolve_download_url(\'models-3.2.0\', \'yolo_11m_nsfw.onnx\'),
	# 				\'path\': resolve_relative_path(\'../.assets/models/yolo_11m_nsfw.onnx\')
	# 			}
	# 		},
	# 		\'size\': (640, 640)
	# 	}
	# }
	return {} # Return empty dict as model is removed


# def get_inference_pool() -> InferencePool: # Commented out unused function
# 	model_names = [ \'yolo_nsfw\' ]
# 	model_source_set = get_model_options().get(\'sources\')
#
# 	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


# def clear_inference_pool() -> None: # Commented out unused function
# 	model_names = [ \'yolo_nsfw\' ]
# 	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	# return create_static_model_set(\'full\').get(\'yolo_nsfw\') # Commented out original return
	return {} # Return empty dict


def pre_check() -> bool:
	# model_hash_set = get_model_options().get(\'hashes\') # Commented out
	# model_source_set = get_model_options().get(\'sources\') # Commented out
	#
	# return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set) # Commented out
	return True # Always return True as checks are disabled


def analyse_stream(vision_frame : VisionFrame, video_fps : Fps) -> bool:
	# global STREAM_COUNTER
	#
	# STREAM_COUNTER = STREAM_COUNTER + 1
	# if STREAM_COUNTER % int(video_fps) == 0:
	# 	return analyse_frame(vision_frame)
	return False # Always return False


def analyse_frame(vision_frame : VisionFrame) -> bool:
	# nsfw_scores = detect_nsfw(vision_frame) # Commented out
	#
	# return len(nsfw_scores) > 0 # Commented out
	return False # Always return False


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
	# vision_frame = read_image(image_path) # Commented out
	# return analyse_frame(vision_frame) # Commented out
	return False # Always return False


@lru_cache(maxsize = None)
def analyse_video(video_path : str, trim_frame_start : int, trim_frame_end : int) -> bool:
	# video_fps = detect_video_fps(video_path) # Commented out
	# frame_range = range(trim_frame_start, trim_frame_end) # Commented out
	# rate = 0.0 # Commented out
	# total = 0 # Commented out
	# counter = 0 # Commented out
	#
	# with tqdm(total = len(frame_range), desc = wording.get(\'analysing\'), unit = \'frame\', ascii = \' =\', disable = state_manager.get_item(\'log_level\') in [ \'warn\', \'error\' ]) as progress: # Commented out
	#
	# 	for frame_number in frame_range: # Commented out
	# 		if frame_number % int(video_fps) == 0: # Commented out
	# 			vision_frame = read_video_frame(video_path, frame_number) # Commented out
	# 			total += 1 # Commented out
	# 			if analyse_frame(vision_frame): # Commented out
	# 				counter += 1 # Commented out
	# 		if counter > 0 and total > 0: # Commented out
	# 			rate = counter / total * 100 # Commented out
	# 		progress.set_postfix(rate = rate) # Commented out
	# 		progress.update() # Commented out
	#
	# return rate > 10.0 # Commented out
	return False # Always return False


# def detect_nsfw(vision_frame : VisionFrame) -> List[Score]: # Commented out unused function
# 	nsfw_scores = []
# 	model_size = get_model_options().get(\'size\')
# 	temp_vision_frame = fit_frame(vision_frame, model_size)
# 	detect_vision_frame = prepare_detect_frame(temp_vision_frame)
# 	detection = forward(detect_vision_frame)
# 	detection = numpy.squeeze(detection).T
# 	nsfw_scores_raw = numpy.amax(detection[:, 4:], axis = 1)
# 	keep_indices = numpy.where(nsfw_scores_raw > 0.2)[0]
#
# 	if numpy.any(keep_indices):
# 		nsfw_scores_raw = nsfw_scores_raw[keep_indices]
# 		nsfw_scores = nsfw_scores_raw.ravel().tolist()
#
# 	return nsfw_scores


# def forward(vision_frame : VisionFrame) -> Detection: # Commented out unused function
# 	content_analyser = get_inference_pool().get(\'content_analyser\')
#
# 	with conditional_thread_semaphore():
# 		detection = content_analyser.run(None,
# 		{
# 			\'input\': vision_frame
# 		})
#
# 	return detection


# def prepare_detect_frame(temp_vision_frame : VisionFrame) -> VisionFrame: # Commented out unused function
# 	detect_vision_frame = temp_vision_frame / 255.0
# 	detect_vision_frame = numpy.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
# 	return detect_vision_frame
