# 1D_tully_CTMQC
Python implementation of CTMQC equations  of the 1D tully models

This is in serial and is fairly unoptimised. I've implemented some basic things like linear interpolation between electronic timesteps to make it less frustratingly slow.


## As of the 20/07/2019 the keyword TESTED in the commit message means the code has explicitly passed the following tests:
#### Misc
* Rabi Oscillation (diabatic propagation)
* dx/dt = v
* No crashes (unless they are definitely not caused by a bug... -the cause should be known and documented)

#### Ehrenfest
* Ehrenfest same as in Agostini, 16 -model4 is a problem this may not be correct (adiabatic and diabatic propagation)
* Ehrenfest same as in Gossel, 18 (adiabatic and diabatic propagation)
* Ehrenfest norm conserved to 1e-12 for dt_n=0.1 dt_e=0.01 (adiabatic propagation -diabatic slightly worse)

#### CTMQC
* 1 rep same as Ehrenfest (pass all Ehrenfest tests)
* d/dt(f_l) = Fad_l
* sigmal calculator working
* sigma calculator working


