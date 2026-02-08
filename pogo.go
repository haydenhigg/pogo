package pogo

import (
	"maps"
	"math"
	"slices"
	"fmt"
)

type X = map[string]float64
type Optimizer struct {
	x X
	space Space
}

// initializer
func New(space Space) *Optimizer {
	return &Optimizer{
		x:     RandomSample(space),
		space: space,
	}
}

// setter
func (opt *Optimizer) Set(k string, v float64) *Optimizer {
	if _, ok := opt.x[k]; ok {
		if bounds, ok := opt.space[k]; ok {
			opt.x[k] = bounds.Clip(v)
		} else {
			opt.x[k] = v
		}
	}

	return opt
}

// getter
func (opt *Optimizer) X() X {
	x := make(X, len(opt.x))
	maps.Copy(x, opt.x)

	return x
}

// optimizers
type Observation struct {
	Input  X
	Output float64
}

func (a *Observation) Compare(b *Observation) int {
	if a.Output < b.Output {
		return -1
	} else if a.Output > b.Output {
		return 1
	} else {
		return 0
	}
}

type ObjectiveFunc = func(X) float64
type SamplerFunc = func(Space) X

func (opt *Optimizer) TPEMinimize(
	f ObjectiveFunc,
	numCandidates int,
	numEpochs int,
	gamma float64,
) *Optimizer {
	if numCandidates <= 0 {
		return opt
	}

	observations := make([]*Observation, 0, numCandidates + numEpochs)

	for _ = range numCandidates {
		x := RandomSample(opt.space)
		observations = append(observations, &Observation{
			Input: x,
			Output: f(x),
		})
	}

	bestY := f(opt.x)

	for _ = range numEpochs {
		// split observations with gamma
		slices.SortFunc(observations, func(a, b *Observation) int {
			return a.Compare(b)
		})

		// splitIndex := max(int(gamma * float64(len(observations))), 1)
		splitIndex := min(10, len(observations))
		good := observations[:splitIndex]
		bad := observations[splitIndex:]

		if good[0].Output < bestY {
			bestY = good[0].Output
			opt.x = good[0].Input
		}

		// infer bandwidths per dimension
		goodBandwidth := make(map[string]float64, len(opt.space))
		// badBandwidth := make(map[string]float64, len(opt.space))
		for k, bounds := range opt.space {
			width := bounds.Max - bounds.Min
			goodBandwidth[k] = width / float64(min(max(len(observations), 10), 100))
			// goodBandwidth[k] = width / float64(min(max(len(good), 10), 100))
			// badBandwidth[k] = width / float64(min(max(len(bad), 10), 100))
		}

		bestSample := X{}
		bestScore := math.Inf(-1)
		for _ = range numCandidates {
			sample := make(X, len(opt.space))
			score := 1.
			for k, bounds := range opt.space {
				sample[k] = bounds.Clip(kdeSample(goodBandwidth[k], good, k))

				pGood := kdePDF(sample[k], goodBandwidth[k], good, k)
				pBad := kdePDF(sample[k], goodBandwidth[k], bad, k)

				score *= pGood / (pBad + 1e-12)
			}

			if score > bestScore {
				bestScore = score
				bestSample = sample
			}
		}

		y := f(bestSample)
		if y < bestY {
			bestY = y
			opt.x = bestSample
		}

		fmt.Println(y)

		observations = append(observations, &Observation{
			Input: bestSample,
			Output: y,
		})
	}

	return opt
}
