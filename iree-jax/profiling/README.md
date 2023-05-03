venv setup/teardown:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jax
python -m pip install --upgrade "jax[cpu]"
deactivate
```

IR dump:
```bash
mkdir ir
export JAX_DUMP_IR_TO=$PWD/ir
python jax_to_mlir.py
cat ./ir/jax_ir1_jit_selu.mlir
```

possible next steps
  * mnist or imagenet -> jax.jit -> IR -> iree-compile/iree-run-module
  * toy program with multiple functions / lines of python code, check source locs
    * try source locs through iree-compile/iree-run-module and in Tracy

### CNN

```
export JAX_DUMP_IR_TO=$PWD/cnn
python jax_cnn.py
cat $PWD/cnn/jax_ir1_jit_run_cnn.mlir
```


<!-- --iree-hal-dump-executable-sources-to=/usr/local/google/home/scotttodd/code/iree-tmp/mobilebertsquad_vulkan_2023_05_01 -->
```
~/code/iree-build/tools/iree-compile \
  --iree-input-type=mhlo \
  $PWD/cnn/jax_ir1_jit_run_cnn.mlir \
  --iree-hal-target-backends=vulkan-spirv \
  -o ~/code/iree-tmp/jax_cnn_vulkan_2023_05_03.vmfb

TRACY_NO_EXIT=1 ~/code/iree-build/tools/iree-run-module \
  --module=~/code/iree-tmp/jax_cnn_vulkan_2023_05_03.vmfb \
   --device=vulkan \
   --function=main \
   --input=32x64x64x10xf32
```
