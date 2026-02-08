package pogo

import "math/rand"

func RandomSample(space Space) X {
	x := make(X, len(space))
	for k, bounds := range space {
		x[k] = rand.Float64()*(bounds.Max-bounds.Min) + bounds.Min
	}

	return x
}
