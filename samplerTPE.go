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

func (sampler *TPESampler) kdePDF(
	x float64,
	trials Trials,
	k string,
	sigma float64,
) float64 {
	n := trials.N()
	pPrior := normPDF(x, sampler.space[k].Midpoint(), sampler.space[k].Width())

	// if there are no trials, use prior only
	if n == 0 {
		return pPrior
	}

	// calculate empirical probability
	pEmpSum := 0.
	for _, trial := range trials {
		pEmpSum += normPDF(x, trial.Input[k], sigma)
	}

	wPrior := 1 / (n + 1)
	wEmp := 1 - wPrior

	return pPrior*wPrior + (pEmpSum / n)*wEmp
}

func (sampler *TPESampler) kdeSample(
	trials Trials,
	k string,
	sigma float64,
) float64 {
	n := trials.N()
	r := rand.NormFloat64()

	if n == 0 || rand.Float64() < 1 / (n + 1) {
		return r*sampler.space[k].Width() + sampler.space[k].Midpoint()
	}

	return r*sigma + trials[rand.Intn(len(trials))].Input[k]
}

func optuna4Bandwidth(domain *Domain, trials Trials, d int) float64 {
	return (domain.Width() / 5) * math.Pow(trials.N(), -1/(float64(d)+4))
}

// "magic" clipping
func minBandwidth(domain *Domain, trials Trials) float64 {
	return domain.Width() / min(100, trials.N())
}

func (sampler *TPESampler) Sample(trials Trials) X {
	if len(trials) < sampler.NumCandidates {
		return randomInSpace(sampler.space)
	}

	// split trial with gamma
	good, bad := trials.Bisected(sampler.Quantile)

	// infer bandwidths
	d := len(sampler.space)

	goodSigma := make(map[string]float64, d)
	badSigma := make(map[string]float64, d)

	for k, domain := range sampler.space {
		minimum := minBandwidth(domain, trials)
		goodSigma[k] = max(optuna4Bandwidth(domain, good, d), minimum)
		badSigma[k] = max(optuna4Bandwidth(domain, bad, d), minimum)
	}

	// sample and evaluate candidates
	bestX := X{}
	bestScore := math.Inf(-1)

	for _ = range sampler.NumCandidates {
		sample := make(X, d)
		score := 1.

		for k, domain := range sampler.space {
			sample[k] = domain.Clip(sampler.kdeSample(good, k, goodSigma[k]))

			pGood := sampler.kdePDF(sample[k], good, k, goodSigma[k])
			pBad := sampler.kdePDF(sample[k], bad, k, badSigma[k])

			score *= pGood / (pBad + 1e-12)
		}

		if score > bestScore {
			bestScore = score
			bestX = sample
		}
	}

	return bestX
}
