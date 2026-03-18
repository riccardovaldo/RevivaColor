# RevivaColor — Deep-Dive Technical Audit Report

> **Reviewer role:** Lead Data Scientist / Academic Reviewer  
> **Audit date:** 2026-03-18  
> **Files audited:** `RevivaColor.ipynb`, `cGANs_implementation.ipynb`, `Utils.py`

---

## 1. Project Objective & Hypothesis

### Core Scientific Question
The project asks: *Can a deep learning model learn the mapping from the luminance channel (L) of a LAB-encoded image to its two chrominance channels (A and B), thereby automatically colorizing a monochromatic photograph of a human subject?*

### Task Framing
This is a **supervised pixel-to-pixel regression** problem, not a classification task. The model receives a single-channel 256 × 256 grayscale image (the L channel) and must produce a two-channel 256 × 256 output (the AB channels) that, when concatenated with the original L channel, reconstructs a plausible full-colour photograph.

### Target Variable
| Variable | Representation | Range after normalisation |
|---|---|---|
| Input `X` | L channel of LAB image | [0, 1] (divided by 100) |
| Output `y` | A and B channels of LAB image | [−1, +1] (divided by 128) |

### Hypothesis
1. A convolutional autoencoder can compress the grayscale signal into a latent representation from which the colour channels can be decoded.
2. Replacing the standard MAE loss with PSNR- or SSIM-based losses—or using Leaky ReLU activations in the decoder—will improve the perceptual quality of the colourisation.
3. A conditional GAN (pix2pix-style) will produce sharper, more realistic colourisation than the autoencoders because the adversarial loss penalises blurry outputs directly.

---

## 2. The Data Pipeline (Step-by-Step Logic)

### 2.1 Data Acquisition & Loading

**Source:** A local `Dataset/` directory on Google Drive containing **4,863 colour JPEG images** of human portraits. The data is not scraped or fetched dynamically; it is assumed to already reside at the path `Dataset/` relative to the working directory.

**Loading function — `Utils.py`, lines 38–58 (`load_images`):**

```python
for file in os.listdir('Dataset'):
    img = np.array(Image.open('Dataset/' + file))
```

The function iterates over every file in the directory using `os.listdir`, opens each one with `PIL.Image`, and converts it to a NumPy array. The resulting array has dtype `float32` (cast explicitly on line 58). The raw pixel values at this stage are in the integer range [0, 255] before any normalisation.

**Four variants of the dataset are loaded** in `RevivaColor.ipynb`, Cell 9:

```python
data         = ut.load_images('Dataset')              # Raw RGB
data_bal     = ut.load_images('Dataset', balance=True)  # White-balanced RGB
data_lab     = ut.load_images('Dataset', lab=True)      # LAB (no balance)
data_lab_bal = ut.load_images('Dataset', balance=True, lab=True)  # White-balanced LAB
```

Only `data` and `data_bal` are used for visual comparison (Cell 11); the actual training pipeline uses `data_lab_bal` exclusively.

**Bug — `folder` parameter is ignored:** The `load_images` signature accepts a `folder` argument, but the function body hard-codes `'Dataset'` on lines 49–50 of `Utils.py` (shown below), silently ignoring the caller's path:
```python
# Utils.py — lines 49–50
for file in os.listdir('Dataset'):               # ← hard-coded, ignores `folder` argument
    img = np.array(Image.open('Dataset/' + file))
```
This means calling `ut.load_images('SomeOtherPath')` silently loads from `'Dataset/'`, making the function's public API misleading.

---

### 2.2 Preprocessing & Cleaning

#### Step 1 — Image Resize (`Utils.py`, line 52)
```python
img = cv2.resize(img, size)   # size=(256, 256) by default
```
All images are resized to 256 × 256 pixels using OpenCV's `cv2.resize`, which applies bilinear interpolation by default. This is necessary because a fully convolutional autoencoder requires a fixed spatial input dimension, and GPU memory limits ruled out higher resolutions (acknowledged in the README).

#### Step 2 — White Balancing (`Utils.py`, lines 22–36)
When `balance=True`, the **Gray World assumption** is applied via `white_balance()`:

```python
img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
avg_a = np.average(img[:, :, 1])   # mean of A channel
avg_b = np.average(img[:, :, 2])   # mean of B channel
img[:, :, 1] -= ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.1)
img[:, :, 2] -= ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.1)
img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
```

The shift applied to channels A and B is proportional to both the deviation of the channel mean from neutral (128) and the local luminance `img[:,:,0] / 255.0`. The scaling factor `1.1` is empirical and not justified theoretically. The operation is performed entirely in LAB space, which is colour-perception-uniform—appropriate for a colour cast correction.

**Note:** This white-balance step is applied only in `RevivaColor.ipynb`. In `cGANs_implementation.ipynb`, Cell 10, the data is loaded **without white balancing**:

```python
data = ut.load_images(DATA_PATH, resize=True, size=(256,256), lab=True)
```

The two notebooks therefore train on slightly different preprocessed data, making a direct numerical comparison of their metrics ambiguous.

#### Step 3 — Colour Space Conversion: RGB → LAB (`Utils.py`, lines 56–57; `skimage.color.rgb2lab`)
```python
if lab:
    img = rgb2lab(img)
```
The image is converted from sRGB to the CIEL\*a\*b\* (LAB) colour space using `skimage.color.rgb2lab`. LAB is perceptually uniform and decouples luminance from chroma, which is the core design choice of the pipeline: the L channel serves as the input (the grayscale signal already present in a black-and-white photo), and the AB channels serve as the prediction target.

After conversion, the LAB channels have the following natural ranges:
- L: [0, 100]
- A: [−128, 127]
- B: [−128, 127]

#### Step 4 — Channel Separation & Normalisation (`RevivaColor.ipynb`, Cell 12)
```python
X = np.array(data_lab_bal)[:,:,:,0] / 100      # L channel → [0, 1]
y = np.array(data_lab_bal)[:,:,:,1:] / 128     # AB channels → [−1, +1]
```
The L channel is divided by 100 to map it to [0, 1]. The AB channels are divided by 128, mapping them to approximately [−1, +1]. This makes the output compatible with the `tanh` activation function used in the final decoder layer, which also outputs in [−1, +1].

**No zero-centering or standard-score normalisation is applied.** The `min_max_prep` function in `Utils.py` (lines 163–179) is defined but **never called** in either notebook.

#### Step 5 — Train / Validation / Test Split (`RevivaColor.ipynb`, Cell 12)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=len(X_test))
```
`train_test_split` from `sklearn.model_selection` is used twice sequentially:
1. **First split:** 10% of the full dataset → test set (≈ 486 images); 90% → training pool.
2. **Second split:** from the training pool, a validation set of the same absolute size as the test set is held out.

This results in approximately:
- Training: ~3,891 images
- Validation: ~486 images
- Test: ~486 images

**No `random_state` is passed** to either `train_test_split` call, making the splits non-reproducible across runs. Despite the seed-setting block at the top of the notebook (Cell 3):
```python
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
```
`sklearn`'s `train_test_split` draws from its own internal random state and ignores `np.random.seed` unless `random_state` is passed explicitly. The fix is `train_test_split(X, y, test_size=0.1, random_state=seed)`. This is a reproducibility flaw.

#### Step 6 — Shape Augmentation (`RevivaColor.ipynb`, Cell 12)
```python
X_train = X_train.reshape(X_train.shape + (1,))
X_val   = X_val.reshape(X_val.shape + (1,))
X_test  = X_test.reshape(X_test.shape + (1,))
```
A trailing channel dimension is added to the L-channel arrays so their shape becomes `(N, 256, 256, 1)`, matching the expected Keras input format `(batch, height, width, channels)`.

#### Step 7 — Test Set Reconstruction for Visualisation (`RevivaColor.ipynb`, Cell 14)
```python
test_pics = np.zeros((X_test.shape[0], 256, 256, 3))
test_pics[:,:,:,0] = X_test[:,:,:,0] * 100
test_pics[:,:,:,1:] = y_test * 128
test_pics = [lab2rgb(img) for img in test_pics]
```
The ground-truth test images are reconstructed in RGB by reversing the normalisation and applying `skimage.color.lab2rgb`. These serve as the visual baseline for qualitative comparison.

---

### 2.3 Feature Engineering

There is **no traditional feature engineering** (no hand-crafted features). The entire feature representation is learned end-to-end by the neural networks. The only domain-motivated design decision is the **choice of the LAB colour space** as the representation, which is itself a form of feature engineering: it decomposes the signal into a component the model receives (L) and a component the model must predict (AB), leveraging the perceptual separability of luminance and chroma.

---

## 3. Modeling Strategy ("The How")

### 3.1 Autoencoder Family (`RevivaColor.ipynb`)

#### Architecture Overview

The autoencoder follows a symmetric encoder–decoder structure built from modular convolutional blocks defined in Cells 20–21.

**Encoder building block — `conv_bn_block` (Cell 20):**
```python
if downsample:
    x = tfkl.MaxPooling2D(...)
for s in range(stack):
    x = tfkl.Conv2D(filters, kernel_size, padding, strides, ...)
    if batch_norm:
        x = tfkl.BatchNormalization(...)
    x = tfkl.Activation(activation, ...)
```
Each block optionally prepends a `MaxPooling2D` (stride 2, 2×2 window) for spatial downsampling, followed by one or more `Conv2D` layers with `BatchNormalization` and an activation (default: `'relu'`). Padding is `'same'` throughout, preserving spatial dimensions within each block.

**Decoder building block — `up_conv_bn_block` (Cell 21):**
```python
for s in range(stack):
    if upsample:
        x = tfkl.UpSampling2D(...)
    x = tfkl.Conv2D(...)
    ...
```
Each block optionally prepends a `UpSampling2D` (factor 2, nearest-neighbour interpolation by default) followed by a `Conv2D`, `BatchNormalization`, and activation. This is a **"resize-convolution"** strategy (as opposed to `Conv2DTranspose`), chosen to avoid checkerboard artefacts.

**Encoder — `build_encoder` (Cell 23):**

| Layer block | Output shape | Notes |
|---|---|---|
| Input | (256, 256, 1) | L channel |
| Block 1 (downsample=True, 8 filters) | (128, 128, 8) | MaxPool + Conv |
| Block 2 (downsample=False, 8 filters) | (128, 128, 8) | Conv only |
| Block 3 (downsample=True, 16 filters) | (64, 64, 16) | MaxPool + Conv |
| Block 4 (downsample=False, 16 filters) | (64, 64, 16) | Conv only |
| Block 5 (downsample=True, 32 filters) | (32, 32, 32) | MaxPool + Conv |
| Block 6 (downsample=False, 64 filters) | (32, 32, 64) | Conv only |
| Block 7 (downsample=True, 128 filters) | (16, 16, 128) | MaxPool + Conv |

The latent representation is a (16, 16, 128) tensor — a 16× spatial compression of the input.

**Decoder — `build_decoder` (Cell 24):**

| Layer block | Output shape | Activation |
|---|---|---|
| Input | (16, 16, 128) | — |
| Block 1 (upsample=True, 128 filters) | (32, 32, 128) | ReLU |
| Block 2 (upsample=False, 64 filters) | (32, 32, 64) | ReLU |
| Block 3 (upsample=True, 32 filters) | (64, 64, 32) | ReLU |
| Block 4 (upsample=False, 16 filters) | (64, 64, 16) | ReLU |
| Block 5 (upsample=True, 16 filters) | (128, 128, 16) | ReLU |
| Block 6 (upsample=False, 8 filters) | (128, 128, 8) | ReLU |
| Block 7 (upsample=True, 2 filters) | (256, 256, 2) | **tanh** |

The final `tanh` activation maps the output to [−1, +1], matching the target AB channel range after normalisation.

**There is no skip connection** between encoder and decoder (unlike U-Net), so the decoder must reconstruct spatial details entirely from the bottleneck.

#### Variant 1 — Autoencoder MAE (Cell 26, 29)
**Instantiation:**
```python
autoencoder.compile(optimizer=tfk.optimizers.AdamW(), loss='mae', metrics=["accuracy"])
```
**Loss:** Mean Absolute Error (MAE), $\mathcal{L}_{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$.  
**Optimiser:** AdamW (Adam with decoupled weight decay regularisation). Default hyperparameters are used (`lr=1e-3`, `weight_decay=1e-4`).  
**Training (Cell 29):**
```python
autoencoder.fit(X_train, y_train, validation_data=(X_val, y_val),
                epochs=200, batch_size=32,
                callbacks=[EarlyStopping(monitor='val_loss', patience=20,
                                         restore_best_weights=True, mode='min')])
```
Early stopping monitors `val_loss` with patience 20. The best weights are restored at the end. Training stops at ≤ 200 epochs.

**Mathematical logic:** MAE is robust to outliers compared to MSE, but it is not differentiable at zero and can lead to underestimation of colour variance (the model tends to predict the mean colour, which is safe but dull — consistent with the blurry reconstructions observed in Cell 37).

#### Variant 2 — Autoencoder PSNR (Cell 40, 41)
**Instantiation:**
```python
autoencoder_psnr.compile(optimizer=tfk.optimizers.AdamW(), loss=PSNRLoss, metrics=["accuracy"])
```
**Loss function (Cell 18):**
```python
def PSNRLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.psnr(y_true, y_pred, 2.0))
```
PSNR is defined as $\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$. Here `max_val=2.0` because the AB channels span [−1, +1] (range of 2). The loss is `1 − mean(PSNR)` so that minimising the loss maximises PSNR.

**Note:** PSNR is based on pixel-level MSE and does not capture perceptual quality. Because it is closely related to MSE, the authors observe (Cell 50) that PSNR-trained results are "almost the same as the ones under the MAE loss function."

#### Variant 3 — Autoencoder SSIM (Cell 52, 53)
**Instantiation:**
```python
autoencoder_ssim.compile(optimizer=tfk.optimizers.AdamW(), loss=SSIMLoss, metrics=["accuracy"])
```
**Loss function (Cell 18):**
```python
def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
```
SSIM measures structural similarity: $\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$. It captures luminance, contrast, and structure jointly, making it more perceptually relevant than MAE or MSE. The observation in Cell 57 confirms that "the distribution of LAB values is more in line with the true ones under the SSIM."

#### Variant 4 — Autoencoder SSIM + Leaky ReLU (Cells 65–67)
The decoder is replaced by `build_decoder_leaky_relu`, defined in Cell 65:
```python
x = up_conv_bn_block(x, 16, 3, upsample=False, activation=tfkl.LeakyReLU(alpha=0.3), name='4')
x = up_conv_bn_block(x, 16, 3, upsample=True,  activation=tfkl.LeakyReLU(alpha=0.3), name='5')
x = up_conv_bn_block(x,  8, 3, upsample=False, activation=tfkl.LeakyReLU(alpha=0.3), name='6')
```
Blocks 4, 5, and 6 of the decoder use `LeakyReLU(alpha=0.3)` instead of `'relu'`. The hypothesis (Cell 64) is that standard ReLU zeroes out negative pre-activations, preventing the filters from learning negative weights in the intermediate layers, which may limit the range of predicted AB values. Leaky ReLU passes a fraction (0.3) of the negative signal through, allowing the network to explore a wider colour range.

#### Variant 5 — Autoencoder Combined Loss + Leaky ReLU (Cells 78–80)
```python
autoencoder_comb_relu.compile(
    optimizer=tfk.optimizers.AdamW(),
    loss=['mae', PSNRLoss, SSIMLoss],
    metrics=["accuracy"]
)
```
All three losses are passed simultaneously to `compile`. Keras sums all losses to produce a single scalar that is backpropagated. The commentary in Cell 77 explicitly states there is **"no real theoretical reason to combine all the losses"**; this is an exploratory experiment.

---

### 3.2 Conditional GAN — pix2pix-Inspired (`cGANs_implementation.ipynb`)

#### Generator — U-Net Architecture (Cell 21)

The generator is a **U-Net** with 6 downsampling blocks and 5 upsampling blocks plus a final transposed convolution:

**Downsampling blocks — `unet_down` (Cell 18):**
```python
x = tfkl.Conv2D(filters, k_size, stride, padding='same', use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.02))(previous_layer)
if batchnorm:
    bn = tfkl.BatchNormalization()(x)
    x = tfkl.LeakyReLU(0.2)(bn)
else:
    x = tfkl.LeakyReLU(0.2)(x)
```
Kernel weights are initialised from a normal distribution N(0, 0.02) as in the original pix2pix paper. The first block omits batch normalisation (as per pix2pix convention). All downsampling uses `stride=2` rather than max-pooling.

**Upsampling blocks — `unet_up` (Cell 19):**
```python
concat = tfkl.Concatenate()([previous_layer, skip])  # skip connection
x = tfkl.Conv2DTranspose(filters, k_size, stride, padding='same', use_bias=False, ...)(concat)
if dropout:
    do = tfkl.Dropout(do_value)(x)
    x = tfkl.ReLU()(do)
```
Skip connections concatenate the encoder feature maps to the corresponding decoder inputs, enabling the generator to reuse fine spatial details. Dropout (rate 0.3) is applied to the first two upsampling blocks (`up1`, `up2`) to act as a stochastic regulariser during training (analogous to the pix2pix paper's approach).

**Generator topology (Cell 21):**

| Block | Output shape | Notes |
|---|---|---|
| Input | (256, 256, 1) | L channel |
| down1 (64 filters, no BN) | (128, 128, 64) | stride=2 |
| down2 (64 filters) | (64, 64, 64) | stride=2 |
| down3 (128 filters) | (32, 32, 128) | stride=2 |
| down4 (256 filters) | (16, 16, 256) | stride=2 |
| down5 (512 filters) | (8, 8, 512) | stride=2 |
| down6 (512 filters) | (4, 4, 512) | stride=2 — bottleneck |
| up1 (512, no skip, dropout) | (8, 8, 512) | |
| up2 (512, skip=down5, dropout) | (16, 16, 1024→512) | skip concat |
| up3 (256, skip=down4, no dropout) | (32, 32, 768→256) | skip concat |
| up4 (128, skip=down3, no dropout) | (64, 64, 384→128) | skip concat |
| up5 (64, skip=down2, no dropout) | (128, 128, 192→64) | skip concat |
| last (concat down1+up5, Conv2DTranspose 2 filters) | (256, 256, 2) | no tanh |

The final `Conv2DTranspose` (Cell 21, `name='last'`) outputs raw logits — **there is no tanh activation on the generator output**. This is important because the discriminator uses `from_logits=True` (Cell 16), and the generator output is passed directly to the discriminator and to the L1 loss without post-hoc clamping during training. For visualisation (Cells 46, 61), the raw output is multiplied by 128 and added back to the L channel, which may produce AB values outside the nominal [−128, 127] range.

#### Discriminator — PatchGAN (Cell 25)

The discriminator is a **PatchGAN** with a 16×16 receptive field:
```python
input  = tfkl.Input(INPUT_GEN)   # (256, 256, 1) — L channel
target = tfkl.Input(OUTPUT)      # (256, 256, 2) — AB channels
concat = tfkl.Concatenate()([input, target])  # (256, 256, 3)
down1  = unet_down(concat, 64, batchnorm=False, stride=2)   # (128, 128, 64)
down2  = unet_down(down1, 128, batchnorm=True, stride=2)    # (64, 64, 128)
last   = Conv2D(1, kernel_size=4, strides=1)(down2)         # (61, 61, 1)
```
The discriminator receives the **concatenation** of the L channel with either the ground-truth AB channels (real) or the generator's AB prediction (fake). It produces a (61, 61, 1) map of real/fake logits — one scalar per patch — rather than a single global binary score. This encourages the generator to be locally consistent, not just globally plausible.

#### Loss Functions (Cells 29–31)

**Generator loss:**
```python
gan_loss  = loss_function(tf.ones_like(disc_output), disc_output)  # adversarial
l1_loss   = tf.reduce_mean(tf.abs(target - gen_output))             # L1/MAE
total_gen = gan_loss + LAMBDA * l1_loss                             # LAMBDA=100
```
The adversarial term pushes the generator to fool the discriminator. The L1 term (weight `LAMBDA=100`) anchors the output near the ground truth. The asymmetric weighting (100:1 of L1 vs. GAN) means the model is primarily MAE-driven, with the adversarial signal providing a sharpening correction. `loss_function = BinaryCrossentropy(from_logits=True)`.

**Discriminator loss:**
```python
loss_1s = loss_function(tf.ones_like(disc_real_output), disc_real_output)
loss_0s = loss_function(tf.zeros_like(disc_fake_output), disc_fake_output)
total   = loss_1s + loss_0s
```
Standard non-saturating GAN loss: real images labelled 1, generated images labelled 0.

**Optimisers (Cell 31):**
```python
generator_optimizer     = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)
```
`lr=2e-4` and `beta_1=0.5` are the exact values recommended in the original pix2pix paper (Isola et al., 2017).

**Batch size:** `BATCH_SIZE=1` (Cell 15), consistent with the pix2pix paper's finding that batch size 1 works best for the instance normalisation / batch normalisation regime in U-Nets.

#### Custom Training Loop (Cell 34–35)

The training loop is **fully custom** (not Keras `model.fit`):
```python
@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real  = discriminator([input_image, target], training=True)
        disc_fake  = discriminator([input_image, gen_output], training=True)
        gen_total, gen_gan, gen_l1 = generator_loss(disc_fake, gen_output, target)
        disc_loss = discriminator_loss(disc_real, disc_fake)
    gen_grads  = gen_tape.gradient(gen_total, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return gen_gan, gen_l1, gen_total, disc_loss
```
`tf.GradientTape` records operations for both networks simultaneously, and then gradients are applied separately. The `@tf.function` decorator compiles the training step into a TensorFlow graph for performance. The `fit` function in Cell 35 uses `train.repeat().take(steps)` to iterate for `epochs × len(train)` steps, computing and logging epoch-level averaged losses.

**Training was interrupted** (Cell 38 documents a Colab failure at epoch 110 out of 150). The model was restored from checkpoint `ckpt-11` (Cell 39) and training continued for an additional 40 epochs (Cell 40), totalling approximately 150 epochs but with a discontinuity at step 110.

---

## 4. Evaluation & Interpretation

### 4.1 Metrics

Three metrics are computed on the **test set** for all autoencoder variants (Cells 107–110) and the cGAN (Cell 47), all computed on the **normalised AB channel predictions** (range [−1, +1], `max_val=2.0`):

| Metric | Formula | Implementation |
|---|---|---|
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | `tf.reduce_mean(tf.abs(y_test - pred))` |
| SSIM | Structural Similarity Index | `tf.reduce_mean(tf.image.ssim(y_test, pred, max_val=2.0))` |
| PSNR | $10\log_{10}\!\left(\frac{4}{\text{MSE}}\right)$ dB | `tf.reduce_mean(tf.image.psnr(y_test, pred, max_val=2.0))` |

The `accuracy` metric passed to `compile` is **meaningless for a regression task** — Keras computes it as element-wise equality, which will almost always be 0 for floating-point predictions.

### 4.2 Validation Logic

**Autoencoder:** Standard holdout split (see Section 2.2, Step 5). `EarlyStopping(patience=20)` acts as implicit regularisation by halting training when validation loss stops improving.

**cGAN:** The same holdout split is used. However, no early stopping is implemented in the custom training loop; the model is trained for a fixed 150 epochs (with the checkpoint restart described above). Checkpoints are saved every 10 epochs (Cell 35).

**No cross-validation** is used in either notebook, which is acceptable for large image datasets where K-fold would be computationally prohibitive.

### 4.3 Results

The exact numerical results were generated in an external Colab session and are not embedded in the notebook output cells in this repository. Based on the code and the commentary left in markdown cells, the qualitative findings are:

| Model | Qualitative visual result | Author commentary |
|---|---|---|
| Autoencoder MAE | Blurry, limited colour range | Cell 37: "reconstructions look very blurry ... predicted values have a limited range" |
| Autoencoder PSNR | Similar to MAE | Cell 50: "almost the same as the ones under the MAE" |
| Autoencoder SSIM | Better colour distribution | Cell 57: "distribution of LAB values is more in line with the true ones" |
| Autoencoder SSIM + Leaky ReLU | Best autoencoder variant; occasional red artefacts | Cell 76: "even better results but with some red stains" |
| Autoencoder Combined Loss + Leaky ReLU | Blurred | Cell 89: "imprecise and blurred reconstructions" |
| cGAN | Good on training set, struggles on test set | Cell 57: "fails to be precise in challenging contexts with numerous people" |

The README states: "Autoencoders slightly outperformed the cGAN across all metrics used for evaluation, including PSNR and SSIM."

---

## 5. Critical Critique

### 5.1 Logical Gaps & Missing Pipeline Steps

#### No data augmentation
Neither notebook applies any augmentation (horizontal flips, random crops, brightness jitter). For a dataset of ~4,863 images trained for 200 epochs, the models see each image approximately 200 times with the same exact pixel values. This is a significant missed opportunity to improve generalisation, particularly for the cGAN which the authors acknowledge needs more data.

#### `load_images` ignores the `folder` argument (`Utils.py`, line 49)
```python
for file in os.listdir('Dataset'):   # ← hard-coded, ignores `folder` parameter
    img = np.array(Image.open('Dataset/' + file))
```
The function signature `def load_images(folder, ...)` accepts a path, but the body always reads from `'Dataset/'`. This silently breaks any attempt to use the function with a different directory.

#### `train_test_split` called without `random_state` (`RevivaColor.ipynb`, Cell 12; `cGANs_implementation.ipynb`, Cell 13)
Despite setting `np.random.seed(42)` and `random.seed(42)`, the `random_state` parameter of `train_test_split` is not passed. The split will differ between Python sessions because `sklearn` uses its own random state. This makes results non-reproducible.

#### White balancing applied only in autoencoder notebook
`data_lab_bal` (white-balanced) is used in `RevivaColor.ipynb`, but the cGAN notebook loads data without `balance=True`. This inconsistency means the two architectures are not evaluated under equivalent data conditions, invalidating the direct metric comparison.

#### `min_max_prep` is defined but never called
`Utils.py` lines 163–179 define a min-max normalisation function with proper train-statistics-only fitting, but it is never invoked. Instead, manual division by constants (100 and 128) is used. The constant normalisation is correct but the utility function is dead code.

#### `plot_history` references `'accuracy'` key (`Utils.py`, lines 92–95)
```python
ax1.plot(history['accuracy'], ...)
ax1.plot(history['val_accuracy'], ...)
```
For regression tasks (MAE, PSNRLoss, SSIMLoss losses), the Keras history dictionary does not contain an `'accuracy'` key unless explicitly added via `metrics=["accuracy"]`. While the notebooks do pass `metrics=["accuracy"]`, this metric is arithmetically meaningless for pixel-value regression (it counts exact floating-point matches) and its inclusion in training plots is misleading.

#### No activation on the generator's final layer
The generator in `cGANs_implementation.ipynb` (Cell 21) ends with a raw `Conv2DTranspose` with no activation. The AB channel targets are normalised to [−1, +1], and the generator output is unbounded. During inference (Cell 46), the predictions are multiplied directly by 128 and used as LAB AB values, potentially producing values outside the valid LAB range.

#### Training was interrupted (cGAN, Cell 38)
Colab OOM/timeout terminated training at epoch 110. The model was restored from `ckpt-11` (representing epoch 110) and trained for an additional 40 epochs. The loss history from the first 110 epochs was lost. The `epoch_losses` dictionary (Cell 35) is reset for the second training call, so no continuous epoch-loss curve is available for the cGAN.

#### `plot_images` indexing bug (`Utils.py`, line 70)
```python
ax = axes[i%2, i%num_img//2]
```
Due to Python operator precedence, `i%num_img//2` is evaluated as `(i%num_img)//2`, which is equivalent to `i//2` for `i < num_img`. For `num_img=10`, axes has shape `(2, 5)`, so the indexing produces valid results coincidentally. However, if `num_img` is not a multiple of 2 or the figure is resized, this will produce an `IndexError`.

### 5.2 Primary Limitations of the Current Implementation

1. **Limited dataset size:** 4,863 images at 256×256 is small for the generative task. The pix2pix paper used ~1,000–10,000+ images for comparable tasks, and the authors acknowledge the cGAN likely needs more data.

2. **No perceptual loss / pre-trained feature backbone:** All autoencoders compute loss directly in pixel/channel space. A VGG-based perceptual loss (as in Johnson et al., 2016) would penalise structural differences in feature space and likely reduce blurriness.

3. **No skip connections in autoencoder:** The autoencoder lacks skip connections. The 16×16 bottleneck discards fine-grained spatial information. Adding skip connections (as in U-Net) — which the cGAN does use — would allow the decoder to reconstruct edges and textures from encoder feature maps.

4. **Image resolution capped at 256×256:** GPU RAM constraints prevented training at higher resolutions. Colour details, especially skin tones, are lost at this resolution.

5. **Metrics are computed on AB channels only, not on full RGB images:** All MAE, SSIM, and PSNR values are calculated on the normalised [−1, +1] AB predictions (2-channel). No metric is computed on the final 3-channel RGB reconstruction, making it difficult to correlate the metric values with the visual quality of the colourised images.

6. **`accuracy` metric is meaningless for regression:** Keras's `accuracy` computes element-wise equality, which will approach 0 for all floating-point predictions. Using `MeanSquaredError` or `MeanAbsoluteError` as the monitored metric would provide a more informative training signal in the Keras callback.

7. **cGAN evaluation is incomplete:** The cGAN training curve is unavailable (Colab failure, Cell 43). No validation loss curve exists to assess convergence or overfitting. The train-set visual results in Cell 50–51 show strong performance that the authors attribute to possible overfitting, but no quantitative evidence (e.g., the gap between train and test MAE) is reported.
