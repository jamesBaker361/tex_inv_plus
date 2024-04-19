TEXT_INPUT_IDS="text_input_ids"
CLIP_IMAGES='clip_images'
IMAGES="images" #in text_to_image_lora this is aka pixel_values
PRIOR_IMAGES="prior_images"
PRIOR_TEXT_INPUT_IDS="prior_text_input_ids"
TEXT_PROMPT="text_prompt"
PLACEHOLDER="<S>"

VANILLA="vanilla"
T5_UNET="t5_unet"
T5_TRANSFORMER="t5_transformer"
LLAMA_UNET="llama_unet"

#token strategy
DEFAULT="default"
MULTI="multi"
THIRD="third"
HALF="half"

#metrics:
PROMPT_SIMILARITY="prompt_similarity"
IDENTITY_CONSISTENCY="identity_consistency"
TARGET_SIMILARITY="target_similarity"
AESTHETIC_SCORE="aesthetic_score"
IMAGE_REWARD="image_reward"

METRIC_LIST=[PROMPT_SIMILARITY, IDENTITY_CONSISTENCY, TARGET_SIMILARITY, AESTHETIC_SCORE, IMAGE_REWARD]

NEGATIVE="over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"