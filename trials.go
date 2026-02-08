package poe

import "slices"

type Trial struct {
	Input  X
	Output float64
}

func NewTrial(f ObjectiveFunc, x X) *Trial {
	return &Trial{Input: x, Output: f(x)}
}

func (a *Trial) Compare(b *Trial) int {
	if a.Output < b.Output {
		return -1
	} else if a.Output > b.Output {
		return 1
	} else {
		return 0
	}
}

type Trials []*Trial

func (trials Trials) Insert(trial *Trial, direction int) Trials {
	low, high := 0, len(trials)
	for low < high {
		mid := (low + high) / 2
		if trials[mid].Compare(trial) == direction {
			low = mid + 1
		} else {
			high = mid
		}
	}

	return slices.Insert(trials, low, trial)
}

func (trials Trials) Bisected(quantile float64) (Trials, Trials) {
	if len(trials) <= 1 {
		return trials, Trials{}
	}

	index := max(int(float64(len(trials)) * quantile), 1)

	return trials[:index], trials[index:]
}
