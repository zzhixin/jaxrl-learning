### Prepare the environment

```bash
cd jaxrl-learning
uv venv
uv pip install -r requirements.txt & uv pip install -e ./vendor/gymnax  --config-settings editable_mode=strict
```