---
comments: true
author: krishan
layout: post
categories: python, jax, pax
title: Pax Layer Basics
description: This lab describes the basics for authoring a new Pax layer. 
---

In this blog we are going to use a new library called praxis to create neural networks.

**Pax Layer basics**
- A Pax Layer represents an arbitrary function - possibly - with trainable parameters.
- Have one or more fields and a __call__ method. Additional setup method to initialize variables and create child layers from templates.
- 3 groups of fields.
  - Hyperparameters
    - Data class syntax. Required to have default values
    - frozen when the layer is instantiated.<br>
      `<name>: type = <default value>`
  - Child layers
    - Can be declared <br>
        `<name>: <type> = instance_field(<factory>)`
    
    - Factory is layer class name like Bias or factory   function returning layer
    - From templates: `self.create_child`
  - Layer templates
    
    Declare the template as dataclass element. 
`<name>: fdl.Config[<type>] = template_field(<factory>)`


**`__call__`**
- Similar to forward function
- self.theta.w
  
  refers to trainable weight w  
- `self.get_var(‘var name’)`
  
  refers to non-trainable weights.
- Trainable weights are immutable in `__call__` while non-trainable weights can be updated with 

  `self.update_var('moving_mean', new_moving_mean)`.

- Pax BaseLayer handles key splitting - 
  
  `self.get_next_prng_key()`
- `self.add_summary`
  report summaries to be shown in tensorboard.



```python
import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import layers
```

# Create linear transformation layer


```python
class LinearLayer(base_layer.BaseLayer):
  # hyper params
  input_shape:int = 10
  output_shape:int = 1

  def setup(self) -> None:
    self.create_variable(
        'w',
        praxis.base_layer.WeightHParams(
            shape = ( self.input_shape, self.output_shape),
        
        dtype = jnp.float32,
        init = base_layer.WeightInit.GaussianSqrtFanOut()
        )
    )
  
  def __call__(self, input_arr:jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(input_arr, self.theta.w)
```


```python
ll = LinearLayer(input_shape=10, output_shape=1)
x = jnp.ones((1, 10))
params = ll.init(jax.random.PRNGKey(0), x)
print(ll)
print(params)
ll.apply(params, x)
```

    LinearLayer(
        # attributes
        dtype = float32
        fprop_dtype = float32
        params_init = WeightInit(method='xavier', scale=1.000001)
        skip_lp_regularization = None
        ici_mesh_shape = None
        dcn_mesh_shape = None
        contiguous_submeshes = None
        mesh_axis_names = None
        shared_weight_layer_id = None
        weight_split_dims_mapping = <PaxConfig[BaseLayer.WeightSharding()]>
        activation_split_dims_mapping = <PaxConfig[BaseLayer.ActivationSharding()]>
        input_shape = 10
        output_shape = 1
    )
    {'params': {'w': Array([[ 0.72848314],
           [ 0.02715359],
           [-0.5122817 ],
           [ 0.7711007 ],
           [-0.8339252 ],
           [-1.7766718 ],
           [ 0.47857362],
           [ 1.4492182 ],
           [-0.05934022],
           [ 0.64895594]], dtype=float32)}}





    Array([[0.92126644]], dtype=float32)



# Create Bias Layer


```python
class BiasLayer(base_layer.BaseLayer):
  #hyper params
  input_shape: int = 1

  def setup(self) -> None:
    self.create_variable('b',
                         base_layer.WeightHParams(
                             shape = (self.input_shape),
                             dtype = jnp.float32,
                             init = base_layer.WeightInit.Constant(0.0)
                         ))
  def __call__(self, input_arr:jnp.ndarray) -> jnp.ndarray:
    return self.theta.b + input_arr
```


```python
b = BiasLayer(input_shape=(1,))
params = b.init(jax.random.PRNGKey(0), x)
print(b)
print(params)
print(x, b.apply(params,x))
```

    BiasLayer(
        # attributes
        dtype = float32
        fprop_dtype = float32
        params_init = WeightInit(method='xavier', scale=1.000001)
        skip_lp_regularization = None
        ici_mesh_shape = None
        dcn_mesh_shape = None
        contiguous_submeshes = None
        mesh_axis_names = None
        shared_weight_layer_id = None
        weight_split_dims_mapping = <PaxConfig[BaseLayer.WeightSharding()]>
        activation_split_dims_mapping = <PaxConfig[BaseLayer.ActivationSharding()]>
        input_shape = (1,)
    )
    {'params': {'b': Array([0.], dtype=float32)}}
    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]] [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]


# Two layers in one 


```python
class FFNLayer(base_layer.BaseLayer):
  #hyper params
  input_shape: int = 10
  output_shape: int = 1
  # data class objects should be immutable.
  # you don't want to create a mutable default value at the class level
  # since it would be shared across all instances of the class
  # pax_fiddle.template_field will make sure that a PaxConfig[ReLU] is 
  # created instead of actual RELU class when data class is instantiated.
  activation_tpl:pax_fiddle.Config[layers.activations.BaseActivation] \
  = pax_fiddle.template_field(layers.activations.ReLU)


  def setup(self) -> None:
    linear_tpl = pax_fiddle.Config(
        LinearLayer, input_shape=self.input_shape, output_shape=self.output_shape
        )
    self.create_child('linear', params = linear_tpl)
    
    bias_tpl = pax_fiddle.Config(BiasLayer, input_shape=(1,))
    self.create_child('bias', params = bias_tpl)

    self.activation_tpl.dtype = jnp.float32
    self.create_child('activation', params = self.activation_tpl)

  def __call__(self, input_arr:jnp.ndarray) -> jnp.ndarray:
    x = self.linear(input_arr)
    x = self.bias(x)
    x = self.activation(x)
    return x
```


```python
INPUT_DIM = 5
OUTPUT_DIM = 10
ffn_tpl = pax_fiddle.Config(FFNLayer, input_shape=INPUT_DIM, output_shape=OUTPUT_DIM)
ffn = pax_fiddle.instantiate(ffn_tpl)
ffn
```




    FFNLayer(
        # attributes
        dtype = float32
        fprop_dtype = float32
        params_init = WeightInit(method='xavier', scale=1.000001)
        skip_lp_regularization = None
        ici_mesh_shape = None
        dcn_mesh_shape = None
        contiguous_submeshes = None
        mesh_axis_names = None
        shared_weight_layer_id = None
        weight_split_dims_mapping = <PaxConfig[BaseLayer.WeightSharding()]>
        activation_split_dims_mapping = <PaxConfig[BaseLayer.ActivationSharding()]>
        input_shape = 5
        output_shape = 10
        activation_tpl = <PaxConfig[ReLU(
          params_init=<PaxConfig[WeightInit(method='xavier', scale=1.000001)]>,
          weight_split_dims_mapping[#praxis.pax_fiddle.DoNotBuild]=<PaxConfig[BaseLayer.WeightSharding()]>,
          activation_split_dims_mapping[#praxis.pax_fiddle.DoNotBuild]=<PaxConfig[BaseLayer.ActivationSharding()]>)]>
    )




```python
x = jnp.ones((1, INPUT_DIM))
params = ffn.init(jax.random.PRNGKey(1), x)
print(jax.tree_map(lambda param: param.shape, params))
model_output = ffn.apply(params, x)
model_output
```

    {'params': {'bias': {'b': (1,)}, 'linear': {'w': (5, 10)}}}


    Array([[0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
            0.8527815, 0.9642554, 0.       , 0.       ]], dtype=float32)



# Validate correctness


```python
y = jnp.dot(x, params['params']['linear']['w'])
y = y + params['params']['bias']['b']
y = y.at[y<0].set(0)
y
```




    Array([[0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
            0.8527815, 0.9642554, 0.       , 0.       ]], dtype=float32)




```python
assert (model_output == y).all()
```
