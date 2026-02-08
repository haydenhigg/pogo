package poe

type Domain struct { Min, Max float64 }

func NewDomain(bounds [2]float64) *Domain {
	return &Domain{
		Min: min(bounds[0], bounds[1]),
		Max: max(bounds[0], bounds[1]),
	}
}

func (domain *Domain) Clip(v float64) float64 {
	return min(max(v, domain.Min), domain.Max)
}

type Bounds = map[string][2]float64
type Space map[string]*Domain

func NewSpace(bounds Bounds) Space {
	space := make(Space, len(bounds))
	for k, bounds := range bounds {
		space[k] = NewDomain(bounds)
	}

	return space
}
