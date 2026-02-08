package poe

import "math/rand"

type RandomSampler struct {
	space Space
}

func NewRandomSampler(space Space) *RandomSampler {
	return &RandomSampler{space: space}
}

func randomInSpace(space Space) X {
	x := make(X, len(space))
	for k, domain := range space {
		x[k] = rand.Float64()*(domain.Max-domain.Min) + domain.Min
	}

	return x
}

func (sampler *RandomSampler) Sample(_ Trials) X {
	return randomInSpace(sampler.space)
}
