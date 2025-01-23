# mobilenet_v2 Under Age Detector
### A really simple and fast mobilenet network to detect if there is at least an underage person in a generated image

This is a really simple MobileNet v2 model trained using image generated with random prompts through Stable Diffusion WebUI images ([https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)) that aims to recognize if is present at least an under age person in a generated image.

This model aims to allow the creation of Stable Diffusion images generation services with explicit contents without worrying that they may contain minors. 

The model works at 512 x 512 px of resolution and it's pretty precise in the faces age recognition, but it can gives sometimes false positive results in case of an image with only a part of the body of a dressed female adult. Anyway my advice is to always take the positive result as valid and try to generate the image again with a different seed a certain number of times to be safe.
