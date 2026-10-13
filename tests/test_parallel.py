import energizer as ez

group = ez.init_device_group(distributed=True)

print("rank", group.rank, "world size", group.size)

model = ez.Sequential(
    ez.Linear(128, 256, device="mlx"),
    ez.Linear(256, 10, device="mlx"),
)

# Tensor-parallel conversion example
model.layers = (
    ez.shard_linear(model.layers[0], "all-to-sharded", group=group),
    ez.shard_linear(model.layers[1], "sharded-to-all", group=group),
)

x = ez.Tensor.randn(32, 128, device="mlx")
y = model.layers[1](model.layers[0](x))
print(group.rank, y.shape)
