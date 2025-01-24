# EfficientNet Under Age Detector
### A really simple and fast mobilenet network to detect if there is at least an underage person in a generated image

This is a really simple EfficientNet model trained using image generated with random prompts through Stable Diffusion WebUI images ([https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)) that aims to recognize if is present at least an under age person in a generated image.

This model aims to allow the creation of Stable Diffusion images generation services with explicit contents without worrying that they may contain minors. 

The model works at 600 x 600 px of resolution (better for EfficientNet) and it's pretty precise in the faces age recognition, but it can gives sometimes false positive results in case of an image with only a part of the body (without face) of a dressed adult. Anyway my advice is to always take the positive result as valid and try to generate the image again with a different seed a certain number of times to be safe.

## Notes
- At the begin of train.py and eval.py there is a very lazy "device_name" where to set the type of device to use for the training or evaluation (cpu, cuda, mps etc.).
- In train.py you can change the default url of SD WebUI API (by default http://127.0.0.1:7860) and you can also change the random possible prompts in base of your model specializations and "weak points".
- The random ages of subjects are from 1 to 40 years.
- Model trained with 1000 generated images and 100 test images. When I have enough time, I'll try to re-train the model with 2000 images. <b>80% accuracy on test images</b>.

# Test image examples

<table>
	<tr><td><b>Image</b></td><td><b>Real</b></td><td><b>Predicted</b></td></tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/0c9ee5d573473fe578c5616b5d2ff98e.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 30<br>
			Is minor: 0.0
		</td>
		<td>
			It's underage: No<br>
			Probability: 0.002<br>
			Confidence: 99.6%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/1ffedd6bf2e9f64f63b4c15b2bdc5189.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 10<br>
			Is minor: 1.0		
		</td>
		<td>
			It's underage: Yes<br>
			Probability: 1.000<br>
			Confidence: 100.0%		
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/38992a7580e6112dd96bc336cf7b109e.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 6<br>
			Is minor: 1.0		
		</td>
		<td>
			It's underage: Yes<br>
			Probability: 1.000<br>
			Confidence: 100.0%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/b74f2054bc60e7cf5afe8dc1d09ce35d.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 8<br>
			Is minor: 1.0		
		</td>
		<td>
			It's underage: Yes<br>
			Probability: 1.000<br>
			Confidence: 100.0%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/2a8bea9a60f71f25428ca56b0709164b.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 7<br>
			Is minor: 1.0		
		</td>
		<td>
			It's underage: Yes<br>
			Probability: 1.000<br>
			Confidence: 100.0%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/c185019c7b4f5a704f2f3e67870dadd3.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 26<br>
			Is minor: 0.0		
		</td>
		<td>
			It's underage: No<br>
			Probability: 0.007<br>
			Confidence: 98.6%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/d3b4d035023c0383548440a302aa51bd.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 24<br>
			Is minor: 0.0		
		</td>
		<td>
			It's underage: No<br>
			Probability: 0.000<br>
			Confidence: 100.0%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/d43c7ad248db37508e431769d6c02700.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 19<br>
			Is minor: 0.0		
		</td>
		<td>
			It's underage: No<br>
			Probability: 0.000<br>
			Confidence: 99.9%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/5dfa91a0749021a66dbe9fa28016ffa4.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 7<br>
			Is minor: 1.0		
		</td>
		<td>
			It's underage: Yes<br>
			Probability: 1.000<br>
			Confidence: 100.0%
		</td>	
	</tr>
		<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/dfa29a937372e0fa19ecf3e9c4b6e450.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 2<br>
			Is minor: 1.0		
		</td>
		<td>
			It's underage: No<br>
			Probability: 0.110<br>
			Confidence: 78.0%<br>
			<span style="color:red; font-weight:bold;">!! FALSE NEGATIVE !!</span><br>
			<i>(But it could be also due to a badly generated image)</i>
		</td>	
	</tr>
	</tr>
		<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/129d2a834db1cc0a82c88d7582705274.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 30<br>
			Is minor: 0.0		
		</td>
		<td>
			It's underage: Yes<br>
			Probability: 0.596<br>
			Confidence: 19.2%<br>
			<span style="color:red; font-weight:bold;">!! FALSE POSITIVE !!</span><br>
			<i>(Probably due to the lack of the head with which to evaluate the proportion)</i>
		</td>	
	</tr>
</table>