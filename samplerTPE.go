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

func priorGaussian(domain *Domain) (float64, float64) {
	return (domain.Min + domain.Max) / 2, domain.Max - domain.Min
}

func kdePDFWithPrior(x float64, domain *Domain, sigma float64, trials Trials, k string) float64 {
	// Prior is always present; if there are no trials, we fall back to pure prior.
	// Weight follows canonical TPE: 1/(n+1) prior, n/(n+1) empirical.
	n := trials.N()
	priorMu, priorSigma := priorGaussian(domain)
	pPrior := normPDF(x, priorMu, priorSigma)

	if n == 0 {
		return pPrior
	}

	sum := 0.
	for _, trial := range trials {
		sum += normPDF(x, trial.Input[k], sigma)
	}
	pEmp := sum / n

	wPrior := 1.0 / (n + 1.0)
	wEmp := 1.0 - wPrior

	return wEmp*pEmp + wPrior*pPrior
}

func kdeSampleWithPrior(domain *Domain, sigma float64, trials Trials, k string) float64 {
	n := trials.N()

	if n == 0 || rand.Float64() < 1 / (n + 1) {
		priorMu, priorSigma := priorGaussian(domain)
		return rand.NormFloat64()*priorSigma + priorMu
	}

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
	badBandwidths := make(map[string]float64, d)

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
			sample[k] = domain.Clip(kdeSampleWithPrior(domain, goodBandwidths[k], good, k))

			pGood := kdePDFWithPrior(sample[k], domain, goodBandwidths[k], good, k)
			pBad := kdePDFWithPrior(sample[k], domain, badBandwidths[k], bad, k)

			score *= pGood / (pBad + 1e-12)
		}

		if score > bestScore {
			bestScore = score
			bestX = sample
		}
	}

	return bestX
}
