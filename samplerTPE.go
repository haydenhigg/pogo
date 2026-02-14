package poe

import (
	"math"
	"math/rand"
)

type TPESampler struct {
	space              Space
	Quantile           float64
	NumCandidates      int
}

func NewTPESampler(
	space Space,
	quantile float64,
	numCandidates int,
) *TPESampler {
	return &TPESampler{
		space:         space,
		Quantile:      quantile,
		NumCandidates: max(numCandidates, 2),
	}
}

func normPDF(x, mu, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}

	num := math.Exp(-0.5 * math.Pow((x-mu)/sigma, 2))
	den := (sigma * math.Sqrt(2*math.Pi))

	return num / den
}

func kdePDF(x, sigma float64, trials Trials, k string) float64 {
	if len(trials) == 0 {
		return 0
	}

	sum := 0.
	for _, trial := range trials {
		sum += normPDF(x, trial.Input[k], sigma)
	}

	return sum / trials.N()
}

func kdeSample(sigma float64, trials Trials, k string) float64 {
	return rand.NormFloat64()*sigma + trials[rand.Intn(len(trials))].Input[k]
}

func optuna4Bandwidth(domain *Domain, trials Trials, d int) float64 {
	quintile := (domain.Max - domain.Min) / 5
	return quintile * math.Pow(trials.N(), -1/(float64(d)+4))
}

// "magic" clipping
func minBandwidth(domain *Domain, trials Trials) float64 {
	return (domain.Max - domain.Min) / min(100, trials.N())
}

func (sampler *TPESampler) Sample(trials Trials) X {
	if len(trials) < sampler.NumCandidates {
		return randomInSpace(sampler.space)
	}

	// split trial with gamma
	good, bad := trials.Bisected(sampler.Quantile)

	// infer bandwidths
	d := len(sampler.space)
	goodBandwidths := make(map[string]float64, d)
	badBandwidths := make(map[string]float64,d)

	for k, domain := range sampler.space {
		minimum := minBandwidth(domain, trials)
		goodBandwidths[k] = max(optuna4Bandwidth(domain, good, d), minimum)
		badBandwidths[k] = max(optuna4Bandwidth(domain, bad, d), minimum)
	}

	// sample and evaluate candidates
	bestX := X{}
	bestScore := math.Inf(-1)

	for _ = range sampler.NumCandidates {
		sample := make(X, d)
		score := 1.

		for k, domain := range sampler.space {
			sample[k] = domain.Clip(kdeSample(goodBandwidths[k], good, k))

			pGood := kdePDF(sample[k], goodBandwidths[k], good, k)
			pBad := kdePDF(sample[k], badBandwidths[k], bad, k)

			score *= pGood / (pBad + 1e-12)
		}

		if score > bestScore {
			bestScore = score
			bestX = sample
		}
	}

	return bestX
}
