
from diffusers import StableDiffusionPipeline

class VanillaPipeline(StableDiffusionPipeline):
    def run_safety_checker(self, image, device, dtype):
        return image, None