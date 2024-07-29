#%%
import subprocess
import sys
import os
from PIL import Image
import numpy as np
from basicpy import BaSiC
from matplotlib import pyplot as plt

os.environ["JAX_PLATFORM_NAME"] = "cpu"

def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running command: {command}\n{result.stderr.decode()}")
    else:
        print(result.stdout.decode())

run_command(f"{sys.executable} -m pip uninstall -yq basicpy")
run_command(f"{sys.executable} -m pip install --upgrade -q basicpy jax jaxlib")

def load_tif(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.stack(images)

images = load_tif('Path/To/.tif')

basic = BaSiC(get_darkfield=True, smoothness_flatfield=1)

try:
    basic.fit(images)
except Exception as e:
    print(f"Error during fitting: {e}")

if basic.baseline is None:
    print("Baseline is None")
else:
    print("Baseline is computed:", basic.baseline)

if basic.baseline is not None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    im = axes[0].imshow(basic.flatfield)
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Flatfield")
    im = axes[1].imshow(basic.darkfield)
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("Darkfield")
    axes[2].plot(basic.baseline)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Baseline")
    fig.tight_layout()
    plt.show()
else:
    print("Skipping plot as baseline is None")

images_transformed = basic.transform(images)

def save_as_tiff(image_array, path):
    img = Image.fromarray(image_array)
    img.save(path, format='TIFF')

save_as_tiff(images_transformed[0], 'test_corrected.tiff')

i = 0
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
im = axes[0].imshow(images[i])
fig.colorbar(im, ax=axes[0])
axes[0].set_title("Original")
im = axes[1].imshow(images_transformed[i])
fig.colorbar(im, ax=axes[1])
axes[1].set_title("Corrected")
fig.tight_layout()
plt.show()































# %%
