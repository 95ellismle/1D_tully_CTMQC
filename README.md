# 1D_tully_CTMQC
Python implementation of CTMQC equations  of the 1D tully models

This is in serial and is fairly unoptimised. I've implemented some basic things like linear interpolation between electronic timesteps to make it less frustratingly slow.


## To push to the master branch make sure the code passes the following tests. After testing use the keyword TESTED in the front of the commit message.
#### Misc
* Rabi Oscillation (diabatic propagation)
* dx/dt = v
* No crashes (unless they are definitely not caused by a bug... -the cause should be known and documented)

#### Ehrenfest
* Ehrenfest same as in Agostini, 17 (adiabatic and diabatic propagation)
* Ehrenfest norm conserved to 1e-12 for dt_n=0.1 dt_e=0.01 (adiabatic propagation -diabatic slightly worse)

#### CTMQC
* 1 rep same as Ehrenfest (pass all Ehrenfest tests)
* d/dt(f_l) = Fad_l


