# 1D_tully_CTMQC
Python implementation of CTMQC equations for some 1D tully models.

CTMQC is an alternative to surface hopping and can be used to simulate nonadiabatic systems (systems in which electronic transitions take place). The equations appear as a corrected Ehrenfest method, where traj are coupled together via a Quantum Momentum term.

To use the code set the variable `inputs' at the top of the main.py code (in the root folder) to one of the commented out values (or leave it as custom). If you wished to run a custom simulation change the parameters in the else statement around line 288. To run multiple simulations at once put multiple values into the lists when choosing your parameters.

**The code is not very user-friendly and as it is just for testing my implementation/sandboxing ideas it isn't designed to be. You may need to be confident with python to use it**


## The most recent version of the code should pass the following tests
#### Misc
 - [x] Rabi Oscillation (diabatic propagation)
 - [x] dx/dt = v
 - [x] No crashes (unless they are definitely not caused by a bug... -the cause should be known and documented)

#### Ehrenfest
 - [x] Ehrenfest same as in Agostini, 16 -model4 is a problem this may not be correct (adiabatic propagation)
 - [ ] Ehrenfest same as in Agostini, 16 -model4 is a problem this may not be correct (diabatic propagation)
 - [x] Ehrenfest same as in Gossel, 18 (adiabatic propagation)
 - [ ] Ehrenfest same as in Gossel, 18 (diabatic propagation)
 - [x] Ehrenfest norm conserved to 1e-12 for dt_n=0.1 dt_e=0.01 (adiabatic propagation)
 - [ ] Ehrenfest norm conserved to 1e-12 for dt_n=0.1 dt_e=0.01 (diabatic propagation)
 - [x] Ehrenfest energy is conserved well

#### CTMQC
 - [x] 1 rep same as Ehrenfest (pass all Ehrenfest tests)
 - [x] d/dt(f_l) = Fad_l
 - [ ] sigmal calculator working
 - [x] Divergence of Rlk calculator smoothing gives better norms (RI0 smoothing)
 - [ ] Divergence of Rlk calculator smoothing (extrapolation smoothing)
 - [x] Norm conservation gets better with decreasing timestep
 - [ ] Energy conservation get better with decreasing timestep
 - [x] 1 rep Qlk = 0
 - [x] All reps same Qlk = 0
 - [x] Sigma is rep independent Qlk = hbar/2 sigma^2
 - [x] Equation S26
