package poe

import (
	"math"
	"math/rand"
)

type TPESampler struct {
	space         Space
	Quantile      float64
	NumCandidates int
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
	for _, observation := range trials {
		sum += normPDF(x, observation.Input[k], sigma)
	}

	return sum / float64(len(trials))
}

func kdeSample(sigma float64, trials Trials, k string) float64 {
	return rand.NormFloat64()*sigma + trials[rand.Intn(len(trials))].Input[k]
}

func inferBinWidthSimple(domain *Domain, trials Trials) float64 {
	// bins := min(max(float64(len(trials)), 10), 100)
	bins := min(max(math.Log(float64(len(trials))), 10), 100)
	return (domain.Max - domain.Min) / bins
}

// Infer bin width using Scott's rule. **This doesn't work!**
func inferBinWidth(domain *Domain, trials Trials, k string) float64 {
	if len(trials) <= 1 {
		return (domain.Max - domain.Min) / 10
	}

	sum := 0.
	for _, trial := range trials {
		sum += trial.Input[k]
	}

	n := float64(len(trials))
	mean := sum / n

	variance := 0.
	for _, trial := range trials {
		d := trial.Input[k] - mean
		variance += d * d
	}

	stdDev := math.Sqrt(variance / (n - 1))

	domainWidth := domain.Max - domain.Min
	binWidthDomain := NewDomain([2]float64{domainWidth / 1000, domainWidth / 4})

	return binWidthDomain.Clip(1.06 * stdDev * math.Pow(n, -0.2))
}

func (sampler *TPESampler) Sample(trials Trials) X {
	if len(trials) < sampler.NumCandidates {
		return randomInSpace(sampler.space)
	}

	// split observations with gamma
	good, bad := trials.Bisected(sampler.Quantile)

	// infer bandwidths
	goodBandwidths := make(map[string]float64, len(sampler.space))
	badBandwidths := make(map[string]float64, len(sampler.space))
	for k, domain := range sampler.space {
		// goodBandwidths[k] = inferBinWidth(domain, good, k)
		// badBandwidths[k] = inferBinWidth(domain, bad, k)
		goodBandwidths[k] = inferBinWidthSimple(domain, good)
		badBandwidths[k] = inferBinWidthSimple(domain, bad)
	}

	// sample and evaluate candidates
	bestX := X{}
	bestScore := math.Inf(-1)

	for _ = range sampler.NumCandidates {
		sample := make(X, len(sampler.space))
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
