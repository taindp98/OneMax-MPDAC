## Extended version of DACBench
We extend the [DACBench](https://github.com/automl/DACBench) package by introducing the (1+($\lambda$, $\lambda$))-GA algorithm on the OneMax problem.

Outline the structure of this package.

```plaintext
dacbench/
├── benchmarks/                     
│   └── theory_benchmarks.py        # Contains the class object interface to configure the environment 
├── envs/                           # Contains algorithm environment
│   ├── theory.py                   # The main theory-derived EAs environment
│   ├── ablation_<>_theory.py       # The supplemented environment supporting the ablation studies
├── instance_sets/                  # Configuration for the problem instance
│   ├── rls_theory/                 # Instances for RLS and LeadingOnes problem
│   ├── ollga_theory/               # Instances for (1+($\lambda$, $\lambda$))-GA and OneMax problem
```

## Quick Start
We offer a customizable checklist to utilize the extended version of DACBench:

1. Check the algorithm environment classes that execute the optimization process in `envs/theory.py`. For instance, we're using the OLL-GA to optimize the OneMax problem. We need two class objects: one for the algorithm and another for the problem instance.

```
class OneMax(BinaryProblem):
    """
    An individual for OneMax problem.

    The aim is to maximise the number of 1 bits in the string
    """

class OLLGATheoryEnv(AbstractEnv):
    """
    Environment for single-parameter control in (1+(lbd,lbd))-GA
    """

class OLLGAFactTheoryEnv(OLLGATheoryEnv):
    """
    Environment for multi-parameter control in (1+(lbd,lbd))-GA
    The action space representation is in factored structure
    """

class OLLGACombTheoryEnv(OLLGATheoryEnv):
    """
    Environment for multi-parameter control in (1+(lbd,lbd))-GA
    The action space representation is in combinatorial structure
    """
```

2. Integrate the implemented algorithm environment classes into the gym-style abstract class located in `benchmarks/theory_benchmark.py`. This class is responsible for configuring the observation and action spaces.

```
class OLLGATheoryBenchmark(AbstractBenchmark):
    """
    Benchmark for single parameter control in (1+(lbd, lbd))-GA
    """

class OLLGAFactTheoryBenchmark(OLLGATheoryBenchmark):
    """
    Benchmark for multi-parameter control in (1+(lbd, lbd))-GA
    The action space representation is in factored structure
    """

class OLLGACombTheoryBenchmark(OLLGATheoryBenchmark):
    """
    Benchmark for multi-parameter control in (1+(lbd, lbd))-GA
    The action space representation is in combinatorial structure
    """
```

3. Create the configured file for the specific problem instance in the `instance_sets/` directory. For instance, if we're working on OLL-GA, the problem instance should be located in `instance_sets/ollga_theory/om_ollga_<problem-size>_<level>.csv`. This CSV file contains three columns: |instance ID|problem size|initial solution|, particularly the initial solution corresponds to the `level` in the file name.

```
## om_ollga_500_medium.csv

id,size,initObj
0,500,0.5
```

## References

```
@inproceedings{eimer-ijcai21,
  author    = {T. Eimer and A. Biedenkapp and M. Reimer and S. Adriaensen and F. Hutter and M. Lindauer},
  title     = {DACBench: A Benchmark Library for Dynamic Algorithm Configuration},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence ({IJCAI}'21)},
  year      = {2021},
  month     = aug,
  publisher = {ijcai.org},
```