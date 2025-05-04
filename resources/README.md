# This folder contains the trained DDQN checkpoints and runtimes produced from other approaches

Outline the structure of this folder.

```plaintext
resources/
├── ddqn_ckpts/                      
│   └── onemax_n<problem-size>_fmp_as_09998.pt     
├── runtimes/
│   └── comparison_method_runtimes.json     
```

## DDQN checkpoints

Please refer to this notebook [test.ipynb](../notebooks/test.ipynb) to replicate the ERT from these checkpoints.

## Other Approaches
We provide the runtimes of other methods including: `theory`, `irace`, `onefifth`, and `rl_dac`

1. `theory`: theory-derived policy across 13 problem sizes, ranging from 100 to 40000.
2. `irace`: static configuration tuned using irace across 13 problem sizes, ranging from 100 to 40000.
3. `onefifth`: population size (lambda) is self-adjusted based on the one-fifth success rule across 6 problem sizes, ranging from 100 to 2000.
4. `rl_dac`: learned policies using DDQN control population size (lambda) standalone across 6 problem sizes, ranging from 100 to 2000.

```json
## structure of comparison_method_runtimes.json
{
    "theory": {
        100: [],
        500: [],
    },
    "irace": {
        100: [],
        500: [],
    },
    "onefifth": {
        100: [],
        500: [],
    },
    "rl_dac": {
        100: [],
        500: [],
    }
}
```