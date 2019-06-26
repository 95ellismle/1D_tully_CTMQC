import numpy as np


def vel_is_diff_x(runData):
    """
    Will determine whether the velocity is the time-derivative of the position.
    """
    velDiff = np.gradient(runData.allR, runData.ctmqc_env['dt'], axis=0)
    vel = runData.allv
    maxDiff = np.max(velDiff - vel)
    avgDiff = np.mean(velDiff - vel)
    if maxDiff > 1e-4:
        print("max vel diff = ", maxDiff)
        print("avg vel diff = ", avgDiff)
        print("\n\n\nERROR: Vel not same as dx/dt. Check velocity verlet!")
        return False
    if avgDiff > 1e-8:
        print("max vel diff = ", maxDiff)
        print("avg vel diff = ", avgDiff)
        print("\n\n\nERROR: Vel not same as dx/dt. Check velocity verlet!")
        return False
    return True


def adMom_is_diff_F(runData):
    """
    Will check if the adiabatic force == d/dt(adiabatic momentum)
    """
    _, nrep, nstate = runData.allAdMom.shape
    dt = runData.ctmqc_env['dt']

    adFrc = runData.allAdFrc
    adMom = runData.allAdMom
    adFrcDiff = np.gradient(adMom, dt, axis=0)

    diffs = (adFrcDiff - adFrc)[np.abs(adMom) > 0]
    print(np.max(diffs))
    print(np.mean(diffs))

    plt.figure()
    plt.plot(runData.allt, adFrc[:, 0, 0], 'k', label="Ad Force Calc")
    plt.plot(runData.allt, adFrc[:, 1:, 0], 'k')
    plt.plot(runData.allt, adFrc[:, :, 1], 'k')
    plt.plot(runData.allt, adFrcDiff[:, 0, 0], 'g--', label="Ad Force Diff")
    plt.plot(runData.allt, adFrcDiff[:, 1:, 0], 'g--')
    plt.plot(runData.allt, adFrcDiff[:, :, 1], 'g--')
    plt.legend()
    plt.show()

    #adMom = runData.allAdMom
    #adMomDict = {'R%iS%i' % (rep, state): adMom[:, rep, state]
    #             for rep in range(nrep)
    #              for state in range(nstate)}
    #adFrcDict = {'R%iS%i' % (rep, state): adFrc[:, rep, state]
    #             for rep in range(nrep)
    #              for state in range(nstate)}

    #adFrcDiff = {}
    #for rep in range(nrep):
    #    for state in range(nstate):
    #        key = 'R%iS%i' % (rep, state)
    #        mask = np.abs(adMomDict[key]) > 0
    #        adFrcNew = np.gradient(adMomDict, dt
    #        adFrcDiff['R%iS%i' % (rep, state)] = adFrcNew
            

