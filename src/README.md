```
uv run maturin develop --release --features pyo3-extension 
&& command time -v uv run pytest tests/test_train_bpe.py
```
