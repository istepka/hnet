Started with uv but couldn't get it to work. The flash_attn package is super problematic to install. 


Now using conda with:
```
conda create -n hnet python=3.11
```
```
pip install torch=2.7.*
pip install mamba_ssm[causal-conv] --no-build-isolation
pip install flash_attn --no-build-isolation
pip install optree omegaconf 
```


# Before, when using torch 2.9
This worked for flash_attn. I think for the future, always install from a pre-build wheel that matches my local specification.   
```
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl
```



# Run
```
python generate.py \
--model-path /data/hf_cache/hnet_2stage_L/hnet_2stage_L.pt \
--config-path configs/hnet_2stage_L.json
```