### Prepare the environment

```bash
cd jaxrl-learning
uv venv
uv pip install -r requirements.txt & uv pip install -e ./vendor/gymnax  --config-settings editable_mode=strict
```

### Principles
- Make the algos files clean, make sure it includes all the core logic, keep it flat. (like cleanrl) 
- Move all the other parts (buffer, logging, eval, etc.) into other modules.
- Use function factories extensively.
- Concentrate logging. Logging online for single run, logging offline for parallel runs.