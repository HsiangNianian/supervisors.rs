# Pipeline Agent Example

A simulated data processing pipeline that uses the **PipelineAgent** pattern
to transform raw log entries through a series of sequential stages.

## Architecture

```
Raw Log Entry
    |
    v
+-------------------+
|  PipelineAgent    |
|                   |
|  stage: parse     |  -- extract fields from raw text
|  stage: enrich    |  -- add metadata (geo, severity)
|  stage: filter    |  -- drop low-severity entries
|  stage: format    |  -- produce structured output
+-------------------+
    |
    v
Structured Alert
```

## Running

```bash
cd examples/pipeline
python main.py
```

No external dependencies required.
