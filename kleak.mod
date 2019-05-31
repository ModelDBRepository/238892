NEURON {
    SUFFIX kleak
    USEION k READ ek WRITE ik
    RANGE ik 
}

ASSIGNED {
    ik (mA/cm2)
    ek (mV)
    v (mV)
}

PARAMETER {
    gk = 0.1e-3
}


BREAKPOINT {
    ik = gk * (v - ek)
}

INITIAL {
    ik = 0
}
