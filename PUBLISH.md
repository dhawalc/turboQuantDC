# Publishing TurboQuantDC to PyPI

## Test PyPI (recommended first)
```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ turboquantdc
```

## Production PyPI
```bash
twine upload dist/*
```

## After publishing
Users can install with:
```bash
pip install turboquantdc
```
