# POe

<ins>**P**</ins>arameter <ins>**O**</ins>ptimization in Go for non-differentiable and costly-to-run objective functions.

Available:

- Random search
- Tree-structured Parzen Estimator (TPE)

## example

```go
package main

import (
	"fmt"
	"math"
	"github.com/haydenhigg/poe"
)

func main() {
	opt := poe.New(func(x poe.X) float64 {
		// (a - 3)^2 + b - 1
		return math.Pow(x["a"]-3, 2) + x["b"] - 1
	})

	space := poe.NewSpace(poe.Bounds{
		"a": {-5, 5},
		"b": {0, 10},
	})

	opt.Minimize(poe.NewTPESampler(space, 0.1, 20), 100)

	fmt.Println(opt.X()) //=> map[a:3 b:0]
}
```

## api

### `X` / `ObjectiveFunc`

```go
type X = map[string]float64
type ObjectiveFunc = func(X) float64
```

### `Poe`

```go
type Poe struct {
	Objective ObjectiveFunc
	Trials    Trials
}

func New(f ObjectiveFunc) *Poe

func (poe *Poe) X() X // best-so-far input (empty if no trials)

// optimization
type Sampler interface { Sample(Trials) X }

func (poe *Poe) Optimize(direction int, sampler Sampler, numTrials int) *Poe
func (poe *Poe) Minimize(sampler Sampler, numTrials int) *Poe // direction = -1
func (poe *Poe) Maximize(sampler Sampler, numTrials int) *Poe // direction = +1
```

### `Trial`/`Trials`

`Trials` stores the results of all executions of the objective function.

```go
// Trial
type Trial struct {
	Input  X
	Output float64
}

func NewTrial(f ObjectiveFunc, x X) *Trial
func (a *Trial) Compare(b *Trial) int

// Trials
type Trials []*Trial

func (trials Trials) Insert(trial *Trial, direction int) Trials // inserts sorted
func (trials Trials) Bisected(quantile float64) (Trials, Trials)
```

### `Domain` / `Space`

`Domain` is a continuous `[min, max]` range with clipping. `Space` maps parameter name -> `Domain`.

```go
type Domain struct { Min, Max float64 }

func NewDomain(bounds [2]float64) *Domain // order-insensitive
func (domain *Domain) Clip(v float64) float64

type Bounds = map[string][2]float64
type Space map[string]*Domain

func NewSpace(bounds Bounds) Space
```

### `RandomSampler`

Uniform random sampling over a `Space`. Ignores past trials.

```go
type RandomSampler struct {
	// no exported fields
}

func NewRandomSampler(space Space) *RandomSampler
func (sampler *RandomSampler) Sample(_ Trials) X
```

### `TPESampler`

Simple Tree-structured Parzen Estimator sampler. Falls back to random sampling until `len(trials) >= NumCandidates`.

```go
type TPESampler struct {
	Quantile      float64
	NumCandidates int
}

func NewTPESampler(space Space, quantile float64, numCandidates int) *TPESampler
func (sampler *TPESampler) Sample(trials Trials) X
```
