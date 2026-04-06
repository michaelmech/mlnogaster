# GeneticFeatureEngineer (Detailed Technical README)

\---

\---

## Key Systems (Implementation Details)

### 1\. Codex (Evolutionary Memory)

* Stored in `self.codex\_programs`
* Accumulates top programs across eras
* Used for:

  * diversity metrics
  * fitness sharing
  * feature reuse

Selection into codex:

&#x20;   programs.sort(...)
    self.codex\_programs += top\_percentile(programs)


\---

### 2\. Fitness Sharing

Implemented via:

&#x20;   def calc\_shared\_fitness(fitness):
        distances = ...
        neighbors = ...
        return fitness + generosity \* neighbor\_mean


Key idea:

* penalizes crowded regions of solution space
* encourages exploration

\---

### 3\. Aging System

Each program gets:

&#x20;   program.birthdate = era \* generations


Penalty:

&#x20;   fitness -= mortality \* age\_penalty(age)


Where `age\_penalty` is Gaussian-like:

&#x20;   exp(-(age - mean)^2 / variance)


Effect:

* older programs gradually lose dominance
* prevents convergence lock-in

\---

### 5\. Gene Encoding

Programs are converted into numeric sequences:

&#x20;   encode\_gene(program)


* operations mapped to integers
* features mapped to indices
* padded to equal length

Used for:

* diversity computation
* gene statistics
* extinction logic

\---

### 6\. Gene Statistics

Computed per generation:

&#x20;   gene\_scores = mean fitness per gene
    gene\_counts = frequency of gene usage


Stored in:

&#x20;   engineer.run\_details\_\['gene\_scores']
    engineer.run\_details\_\['gene\_counts']


\---

### 7\. Gene Extinction

Triggered when stagnation is detected:

&#x20;   if max\_fitness\_current < max\_fitness\_previous:
        extinction\_counter += 1


When threshold reached:

1. Identify dominant gene:

&#x20;purge\_idx = gene\_counts.idxmax()


2. Remove:

   * operation from function set OR
   * feature from input matrix

Effect:

* forces exploration
* prevents over-reliance on specific operators

\---

### 8\. Impostor Gene (Noise Operator)

Custom operation:

&#x20;   f(x1, x2) = random noise


Behavior:

* acts as adversarial feature
* tests robustness of selection

Safety:

* if >75% of programs use it → automatically disabled

\---

### 9\. Pareto Optimization

Supports multiple objectives:

&#x20;   program.pareto\_fitnesses = (primary, metric1, metric2, ...)


Selection via crowding distance:

&#x20;   distance += (neighbor\_diff)


Final score:

&#x20;   fitness = 1 / (crowding\_distance + 1)


Effect:

* balances trade-offs (e.g., accuracy vs complexity)

\---

### 10\. Adaptive Evolution

If enabled:

&#x20;   diversity = calc\_diversity()


Then:

&#x20;   p\_mutation = 1 - diversity / max\_diversity
    n\_participants = scaled(diversity)


Effect:

* low diversity → more mutation
* high diversity → stronger selection

\---

### 11\. Diversity Metrics

Two independent measures:

#### Genetic Diversity

* computed on encoded genes
* distance metric: Jaro-Winkler

#### Phenotypic Diversity

* computed on fitness values
* distance metric: Euclidean

Final:

&#x20;   diversity = superficiality \* genetic + phenotypic


\---

### 12\. Program Pruning

Greedy simplification:

&#x20;   remove node → recompute fitness
    if fitness improves → keep removal


Effect:

* reduces program complexity
* acts as local optimization

\---

### 14\. Feature Space Growth

Each era expands input:

&#x20;   X\_new = concat(X, engineered\_features)


Optional:

* deduplication via `np.unique`
* batch vs full codex modes

\---

## Key Design Philosophy

This system combines:

* Evolutionary search (GP)
* Ecological pressure (aging, seasons, extinction)
* Meta-learning (multi-era accumulation)
* Structural introspection (gene encoding)
* Multi-objective optimization (Pareto fronts)

\---

## When This Excels

* Nonlinear feature discovery
* Unknown feature interactions
* Automated feature engineering
* Problems where manual feature design is difficult

\---

## Tradeoffs

* High computational cost
* Many interacting hyperparameters
* Harder interpretability vs simpler models
* Risk of overengineering

\---

## Summary

This implementation is best understood as:

&#x20;   Evolutionary Feature Engineering System


rather than a traditional GA.

It blends genetic programming with ecological and adaptive mechanisms to continuously expand and refine the feature space.

