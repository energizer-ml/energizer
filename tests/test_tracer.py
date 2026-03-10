import pytest
from energizer.autograd import Tensor
from energizer.coreml.tracer import Tracer, TraceData

def test_tracing_basic_mlp():
    with Tracer() as tracer:
        # Create trace-friendly Tensors
        x = Tensor(TraceData((32, 784), "float32"))
        w1 = Tensor(TraceData((784, 128), "float32"), requires_grad=True)
        b1 = Tensor(TraceData((128,), "float32"), requires_grad=True)
        w2 = Tensor(TraceData((128, 10), "float32"), requires_grad=True)
        b2 = Tensor(TraceData((10,), "float32"), requires_grad=True)
        
        # Manual Forward pass
        h = x @ w1
        h = h + b1
        h = h.maximum(0.0)
        out = h @ w2
        out = out + b2

    nodes = tracer.nodes
    
    # Check that ops were recorded
    assert len(nodes) == 5
    
    # 1. MatMul
    assert nodes[0].op == "MatMul"
    assert nodes[0].output_shape == (32, 128)
    
    # 2. Add
    assert nodes[1].op == "Add"
    assert nodes[1].output_shape == (32, 128)
    
    # 3. Maximum (ReLU)
    assert nodes[2].op == "Maximum"
    assert nodes[2].output_shape == (32, 128)
    
    # 4. MatMul
    assert nodes[3].op == "MatMul"
    assert nodes[3].output_shape == (32, 10)
    
    # 5. Add
    assert nodes[4].op == "Add"
    assert nodes[4].output_shape == (32, 10)

def test_tracing_shape_inference():
    with Tracer() as tracer:
        x = Tensor(TraceData((10, 20), "float32"))
        
        # SumAxis
        s = x.sum(axis=0)
        
        # Transpose
        t = x.T
        
        # Reshape
        r = x.reshape((5, -1))
        
        # Squeeze
        sq = r.reshape((5, 40, 1)).squeeze(axis=2)
        
    nodes = tracer.nodes
    assert len(nodes) == 5
    assert nodes[0].op == "SumAxis"
    assert nodes[0].output_shape == (20,)
    
    assert nodes[1].op == "Transpose"
    assert nodes[1].output_shape == (20, 10)
    
    assert nodes[2].op == "Reshape"
    assert nodes[2].output_shape == (5, 40)
    
    # squeeze actually calls reshape first, so nodes[3] is Reshape
    assert nodes[3].op == "Reshape"
    assert nodes[3].output_shape == (5, 40, 1)
    
    assert nodes[4].op == "Squeeze"
    assert nodes[4].output_shape == (5, 40)
