#loc = loc(unknown)
module @jit_broadcast_in_dim {
  func.func public @main(%arg0: tensor<f32> {mhlo.sharding = "{replicated}"} loc(unknown)) -> tensor<32x64x64x10xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<32x64x64x10xf32> loc(#loc2)
    return %0 : tensor<32x64x64x10xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/usr/local/google/home/scotttodd/code/scratch/iree/jax/jax_cnn.py":27:0)
#loc2 = loc("jit(broadcast_in_dim)/jit(main)/broadcast_in_dim[shape=(32, 64, 64, 10) broadcast_dimensions=()]"(#loc1))
