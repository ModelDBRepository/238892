NEURON {
    SUFFIX taccumulation3
    USEION na READ ina WRITE nai
    USEION k READ ik WRITE ki
    RANGE c1
}

ASSIGNED {
    ina (mA/cm2)
    ik (mA/cm2)
    diam (um)
    c1
}

UNITS {
    FARADAY = (faraday) (kilocoulombs)
}

STATE {
    nai (mM)
    ki (mM)
}

BREAKPOINT {
    SOLVE state METHOD derivimplicit
}

INITIAL {
    nai = 4.297
    ki = 138.116
}

PROCEDURE rates(v(mV)) {
    c1 = 40. / (FARADAY * diam)
}

DERIVATIVE state {
    rates(v)
    ki' = -c1 * ik
    nai' = -c1 * ina
}

