# mobilenet_v2 Under Age Detector
### A really simple and fast mobilenet network to detect if there is at least an underage person in a generated image

This is a really simple MobileNet v2 model trained using image generated with random prompts through Stable Diffusion WebUI images ([https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)) that aims to recognize if is present at least an under age person in a generated image.

This model aims to allow the creation of Stable Diffusion images generation services with explicit contents without worrying that they may contain minors. 

The model works at 512 x 512 px of resolution and it's pretty precise in the faces age recognition, but it can gives sometimes false positive results in case of an image with only a part of the body (without face) of a dressed adult. Anyway my advice is to always take the positive result as valid and try to generate the image again with a different seed a certain number of times to be safe.

## Notes
- At the begin of train.py and eval.py there is a very lazy "device_name" where to set the type of device to use for the training or evaluation (cpu, cuda, mps etc.).
- In train.py you can change the default url of SD WebUI API (by default http://127.0.0.1:7860) and you can also change the random possible prompts in base of your model specializations and "weak points".
- The random ages of subjects are from 1 to 40 years.

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
			Probability: 0.384<br>
			Confidence: 23.1%
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
			Probability: 0.949<br>
			Confidence: 89.7%		
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
			Probability: 0.994<br>
			Confidence: 98.8%	
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
			Probability: 0.667<br>
			Confidence: 33.4%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/2a8bea9a60f71f25428ca56b0709164b.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 7<br>
			Is minor: 1.0		
		</td>
		<td>
			It's underage: No<br>
			Probability: 0.253<br>
			Confidence: 49.4%<br>
			<span style="color:red; font-weight:bold;">!! FALSE NEGATIVE !!</span>
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
			Probability: 0.071<br>
			Confidence: 85.7%
		</td>	
	</tr>
	<tr>
		<td><img src="https://github.com/cekkr/mobilenet_v2_under-age-detector/blob/main/test_images/d3b4d035023c0383548440a302aa51bd.png?raw=true" style="width:256px;"/></td>
		<td>
			Age: 24<br>
			Is minor: 0.0		
		</td>
		<td>
			It's underage: Yes<br>
			Probability: 0.966<br>
			Confidence: 93.2%<br>
			<span style="color:red; font-weight:bold;">!! FALSE POSITIVE !!</span><br>
			<b>This the "lack of face false positive" case.</b>
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
			Probability: 0.004<br>
			Confidence: 99.2%
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
			Probability: 0.955<br>
			Confidence: 91.0%<br>
		</td>	
	</tr>
</table>