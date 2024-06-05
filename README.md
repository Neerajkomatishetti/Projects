In this guide, we will explore KerasCV's Stable Diffusion implementation, show how to use these powerful performance boosts, and explore the performance benefits that they offer.

To get started, let's install a few dependencies and sort out some imports:


pip install tensorflow keras_cv --upgrade --quiet

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 721.6/721.6 kB 13.5 MB/s eta 0:00:00

import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

Introduction

Unlike most tutorials, where we first explain a topic then show how to implement it, with text-to-image generation it is easier to show instead of tell.

Check out the power of keras_cv.models.StableDiffusion().

First, we construct a model:


model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

By using this model checkpoint, you acknowledge that its usage is subject to the terms of the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE
Next, we give it a prompt:


images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)

Downloading data from https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true
1356917/1356917 [==============================] - 0s 0us/step
Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5
492466864/492466864 [==============================] - 9s 0us/step
Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5
3439090152/3439090152 [==============================] - 63s 0us/step
50/50 [==============================] - 126s 295ms/step
Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5
198180272/198180272 [==============================] - 2s 0us/step
png

Pretty incredible!

But that's not all this model can do. Let's try a more complex prompt:


images = model.text_to_image(
    "cute magical flying dog, fantasy art, "
    "golden color, high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
plot_images(images)

50/50 [==============================] - 15s 294ms/step
png

The possibilities are literally endless (or at least extend to the boundaries of Stable Diffusion's latent manifold).

Wait, how does this even work?
Unlike what you might expect at this point, StableDiffusion doesn't actually run on magic. It's a kind of "latent diffusion model". Let's dig into what that means.

You may be familiar with the idea of super-resolution: it's possible to train a deep learning model to denoise an input image -- and thereby turn it into a higher-resolution version. The deep learning model doesn't do this by magically recovering the information that's missing from the noisy, low-resolution input -- rather, the model uses its training data distribution to hallucinate the visual details that would be most likely given the input. To learn more about super-resolution, you can check out the following Keras.io tutorials:

Image Super-Resolution using an Efficient Sub-Pixel CNN
Enhanced Deep Residual Networks for single-image super-resolution
Super-resolution

When you push this idea to the limit, you may start asking -- what if we just run such a model on pure noise? The model would then "denoise the noise" and start hallucinating a brand new image. By repeating the process multiple times, you can get turn a small patch of noise into an increasingly clear and high-resolution artificial picture.

This is the key idea of latent diffusion, proposed in High-Resolution Image Synthesis with Latent Diffusion Models in 2020. To understand diffusion in depth, you can check the Keras.io tutorial Denoising Diffusion Implicit Models.

Denoising diffusion

Now, to go from latent diffusion to a text-to-image system, you still need to add one key feature: the ability to control the generated visual contents via prompt keywords. This is done via "conditioning", a classic deep learning technique which consists of concatenating to the noise patch a vector that represents a bit of text, then training the model on a dataset of {image: caption} pairs.

This gives rise to the Stable Diffusion architecture. Stable Diffusion consists of three parts:

A text encoder, which turns your prompt into a latent vector.
A diffusion model, which repeatedly "denoises" a 64x64 latent image patch.
A decoder, which turns the final 64x64 latent patch into a higher-resolution 512x512 image.
First, your text prompt gets projected into a latent vector space by the text encoder, which is simply a pretrained, frozen language model. Then that prompt vector is concatenated to a randomly generated noise patch, which is repeatedly "denoised" by the diffusion model over a series of "steps" (the more steps you run the clearer and nicer your image will be -- the default value is 50 steps).

Finally, the 64x64 latent image is sent through the decoder to properly render it in high resolution.

The Stable Diffusion architecture

All-in-all, it's a pretty simple system -- the Keras implementation fits in four files that represent less than 500 lines of code in total:

text_encoder.py: 87 LOC
diffusion_model.py: 181 LOC
decoder.py: 86 LOC
stable_diffusion.py: 106 LOC
But this relatively simple system starts looking like magic once you train on billions of pictures and their captions. As Feynman said about the universe: "It's not complicated, it's just a lot of it!"
