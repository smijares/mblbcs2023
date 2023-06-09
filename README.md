# Scalable Reduced-Complexity Compression of Hyperspectral Remote Sensing Images using Deep Learning
Repository with the models to replicate the results from the "Scalable Reduced-Complexity Compression of Hyperspectral Remote Sensing Images using Deep Learning" paper by Sebastià Mijares i Verdú, Johannes Ballé, Valero Laparra, Joan Bartrina-Rapesta, Miguel Hernández-Cabronero, and Joan Serra-Sagristà, submitted for publication in the MDPI Remote Sensing journal in June 2023.
## What is in this repository
In this repository you will find the architecture developed for the paper, the models we trained to replicate our results, and a framework to run these models in a simple manner. To fully replicate our results, you may want to use the GICI implementations of [JPEG 2000](http://gici.uab.cat/BOI/) and [RKLT](http://gici.uab.cat/GiciApps/rklt.tar.gz) we used. Our tests sets can be found in the [GICI datasets](https://gici.uab.cat/GiciWebPage/datasets.php) site, which include the information on how they were obtained and pre-processed.
## How to replicate our results
The MAIN.py script simplifies the training and testing of networks. In the [GICI webpage](http://gici.uab.cat/GiciWebPage/datasets.php) you can find the test sets we used for AVIRIS and Hyperion data stored in our format (16-bit unsigned integers in raw format). You will also find our pre-trained models in that site. Download these and use the following directory structure:

```
parent
├── mblbcs2022
├── models
│     ├── AVIRIS_1band_1
│     ├── AVIRIS_1band_2
│     ├── AVIRIS_1band_3
│     ├── AVIRIS_1band_4
│     ├── AVIRIS_3band_1
│     ├── AVIRIS_3band_2
│     ├── AVIRIS_3band_3
│     ├── AVIRIS_3band_4
│     ├── Hyperion_1band_1
│     ├── Hyperion_1band_2
│     ├── Hyperion_1band_3
│     ├── Hyperion_1band_4
│     ├── Hyperion_3band_1
│     ├── Hyperion_3band_2
│     ├── Hyperion_3band_3
│     └── Hyperion_3band_4
└── datasets
      ├── AVIRIS_test
      └── Hyperion_test
```

The following command is an example of a test of a model:

```
python3 MAIN.py --input_bands 1 --model_path AVIRIS_1band_2 test --dataset AVIRIS_test
```

Note the architecture script used by default is the one discussed in the paper. This will produce a results .csv file in the models/AVIRIS_1band_2 directory, as well as .png visualisations of a randomly sampled image (one band) and its reconstruction to perform a sanity check.

Observe that, if the input images have more bands than indicated in `--input_bands` a new auxiliary dataset will be created for testing purposes where the images are sliced into clusters of `--input_bands` bands each, and the models will be tested among these.

### What if I only want to compress/decompress some images?

To compress/decompress an image (or selection of images) specifically, you'll need to use the `bands_extractor.py` auxiliary script and the architecure script directly. First, use the `bands_extractor.py` script to extract the band clusters you need from a set in `../datasets`:

```
python3 ./auxiliary/bands_extractor.py --source AVIRIS_test --destination AVIRIS_test_3bands --consecutive_bands 3
```

Then, we can use the architecture script to compress the image we're interested in. We can easily use a glob pattern to identify solely the bands clusters with a given name, such as in the following example:

```
python3 ./hyperprior_adaptive.py --model_path ../models/AVIRIS_3bands_2 compress "../datasets/AVIRIS_test_3bands/f080806t01p00r07*.raw"
```

We can then decompress these images using the following command:

```
python3 ./hyperprior_adaptive.py --model_path ../models/AVIRIS_3bands_2 decompress "../datasets/AVIRIS_test_3bands/f080806t01p00r07*.raw.tfci"
```

## How to train a model from scratch
As described before, use the MAIN.py script with the following directory structure.

```
parent
├── mblbcs2022
├── models
│     └── ...
└── datasets
      └── ...
```

For example you may train a model as:

```
python3 MAIN.py --input_bands 1 --model_path my-model test --dataset AVIRIS_training --epochs 100 --steps_per_epoch 1000 --lambda 0.0001 --num_filters 64 384 --patchsize 256
```

This will train a model for single bands of the images in `datasets/AVIRIS_training` for 100 epochs of 1.000 steps each using 64 filters in the hidden layers and 384 filters in the latent space on 256x256 spatial patches. Observe that, if the input images have more bands than indicated in `--input_bands` a new auxiliary dataset will be created for training purposes where the images are sliced into clusters of `--input_bands` bands each, and the models will be trained among these. The model will be stored in `models/my-model` together with training logs, which you can view using [Tensorboard](https://www.tensorflow.org/tensorboard).

Important note: if training is interrupted mid-training, the training process can be started again from the last epoch checkpoint. However, if a model's training is finished, training cannot continue.
