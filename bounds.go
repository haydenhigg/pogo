package pogo

type Bounds struct {
	Min float64
	Max float64
}

func Bound(a, b float64) *Bounds {
	return &Bounds{Min: min(a, b), Max: max(a, b)}
}

func (bounds *Bounds) Clip(v float64) float64 {
	if v < bounds.Min {
		return bounds.Min
	} else if v > bounds.Max {
		return bounds.Max
	} else {
		return v
	}
}

type Space = map[string]*Bounds
