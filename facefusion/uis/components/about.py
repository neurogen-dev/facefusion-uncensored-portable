import random
from typing import Optional

import gradio

from facefusion import metadata, wording

METADATA_BUTTON : Optional[gradio.Button] = None
ACTION_BUTTON : Optional[gradio.Button] = None


def render() -> None:
	global METADATA_BUTTON
	global ACTION_BUTTON

	action = random.choice(
	[
		{
			'wording': wording.get('about.become_a_member'),
			'url': 'https://subscribe.facefusion.io'
		},
		{
			'wording': wording.get('about.join_our_community'),
			'url': 'https://join.facefusion.io'
		},
		{
			'wording': wording.get('about.read_the_documentation'),
			'url': 'https://docs.facefusion.io'
		}
	])

	METADATA_BUTTON = gradio.Button(
		value = 'FaceFusion Next Portable by Neurogen',
		variant = 'primary',
		link = metadata.get('url')
	)
	ACTION_BUTTON = gradio.Button(
		value = action.get('wording'),
		link = action.get('url'),
		size = 'sm'
	)
	# Add links section
	gradio.Markdown(
		"""
		**Другие Portable сборки:**
		*   [Telegram](https://t.me/neurogen_portablebot?start=prt_KJhS2qaKkFizJB6y)
		*   [Boosty](https://boosty.to/neurogen)
		"""
	)
