package poe

import "slices"

type X = map[string]float64
type ObjectiveFunc = func(X) float64

type Poe struct {
	Objective ObjectiveFunc
	Trials    Trials
}

// initializer
func New(f ObjectiveFunc) *Poe {
	return &Poe{
		Objective: f,
		Trials:    Trials{},
	}
}

// getter
func (poe *Poe) X() X {
	if len(poe.Trials) == 0 {
		return X{}
	}

	return poe.Trials[0].Input
}

// optimization
type Sampler interface { Sample(Trials) X }

func (poe *Poe) Optimize(direction int, sampler Sampler, numTrials int) *Poe {
	// re-sort in case optimization direction is different than it was before
	slices.SortFunc(poe.Trials, func(a, b *Trial) int {
		return -direction * a.Compare(b)
	})

	// run trials
	for _ = range numTrials {
		trial := NewTrial(poe.Objective, sampler.Sample(poe.Trials))
		poe.Trials = poe.Trials.Insert(trial, direction)
	}

	return poe
}

func (poe *Poe) Minimize(sampler Sampler, numTrials int) *Poe {
	return poe.Optimize(-1, sampler, numTrials)
}

func (poe *Poe) Maximize(sampler Sampler, numTrials int) *Poe {
	return poe.Optimize(1, sampler, numTrials)
}
