from feature_lens.utils.load_pretrained import load_model

model = load_model("solu-1l")
print(model)

_, cache = model.run_with_cache("hello, world")
print(cache["hook_embed"].shape)
