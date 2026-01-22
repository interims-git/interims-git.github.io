
# Installation

This has been tested on NVidia RTX 3090 w/ py3.10 pt2.4 and cu11.8, and NVidia RTX 4090 w/ py3.10 pt2.4 and cu12.4

1. Create a conda environment with `conda create -n vsres python=3.10` (pytorch version requires compatibility with `gsplat`; I use py310)
2. Install pytorch (requires compatibility with `gsplat`; I use pt24)
3. Download `gsplat` either via `pip install gsplat` or with wheel (I used `gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl`)
4. Run `pip install -r requirements.txt`
5. Use `bash run.sh SRC NAME ARGS` to run the method
## Texture Stats

For `texture_statistic.py` install
```
python3 -m venv texenv
source texenv/bin/activate
pip install numpy scikit-image matplotlib opencv-python
pip install joblib
```

# Models/Git Branches

`main` : Uses triplanes to predict $\lambda, a, b, s$ all view dependant except $s$, and modelled via spherical harmonics. We the use $a,b,s$ to sample a low and high resolution texture using mip-maping/LERP method and use the equation $c' = \lambda c + (1-\lambda)\Delta c (\cdot)$ where $\Delta c(a,b,s,I_{text}) \rightarrow  \text{Texture Sampling}$. This is our naive base model. Much to explore...

`fully_explicit` : Models the spherical harmonics/1-channel parameters for $\lambda, a, b, s$ as per-gaussian points. Otherwise retains similarity to `main`

`fe_deformation` : I forget but maybe I changed from alphablend function to a deformation function

`no_opacity_param` : Remove the opacity variable and treat every point as having opacity=1. (not efficient implementation)

`canon_loss` : Use canonical images to train the canonical color and geometry components

`res_loss` : Uses residual loss function $I^*_{\text{residuals}} = I^*_{\text{deform}} - I^*_{\text{canon}}$ 

`fully_explicit_nomipmap` : Like `fully_explicit` but no mipmapping is used for texture sampling

`mipmap` : Introduces multi-scale mipmaps to texture sampling

`mipmap_scaleloss` : Equates the norm of 3-D Gaussian scales (stop gradient) to the norm of the 2-D texture scales (had gradients), i.e. $\hat{s_{3D}} - \hat{s_{2D}}$. This should force equivalence between gaussian size and sample size meaning that larger gaussians sample coarser features.

`deform_color` : Using color deformation instead of alpha blending for applying $\Delta c$. 

`post_rasterization` : Compute $c' = c + \lambda \Delta c$ after rasterization. Its significantly faster and supposedly should produce smoother texture samples in rendering as the Mahlanobis dist function smooths texture sample coordinates in image space, rather than texture sampling before hand using mipmaps. Results were not good though...

`outer-mip` : Adds a 1px padding to each texture, where `pad_value=[0,0,0]`. This reduces $c'=c$ and enforces the notion that when samples extend past the texture space, they should not be lit by the IBL.



```

# Data Collection Notes

1. Pretain the splatfacto model (via nerfstudio) on the set of canonical images including those for the relighting videos
2. Extract the initial gaussian model + poses for the relighting cameras (both require ns-generate command within nerfstudio)
3. Generate novel view camera path with nerfstuio
4. Load into current model...



