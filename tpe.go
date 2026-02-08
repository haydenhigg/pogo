package pogo

import (
	"math"
	"math/rand"
)

func normPDF(x, mu, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}

	num := math.Exp(-0.5 * math.Pow((x-mu)/sigma, 2))
	den := (sigma * math.Sqrt(2*math.Pi))

	return num / den
}

func kdePDF(x, sigma float64, observations []*Observation, k string) float64 {
	if len(observations) == 0 {
		return 0
	}

	sum := 0.
	for _, observation := range observations {
		sum += normPDF(x, observation.Input[k], sigma)
	}

	return sum / float64(len(observations))
}

func kdeSample(sigma float64, good []*Observation, k string) float64 {
	return rand.NormFloat64()*sigma + good[rand.Intn(len(good))].Input[k]
}
