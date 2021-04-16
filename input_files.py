inputs = "HighVaryingSigma"
all_sig_min = [0.5]*1000
mass = 2000
rootSaveFold = "."

all_dt = [2.0670687287875/i for i in range(5, 500)]
all_elec_steps = [5] * len(all_dt)
if inputs == "FullGossel":
    print("Carrying out full gossel simulations (ehrenfest and ctmqc for all momenta)")
    numRepeats = 1
    all_nRep = [100] * 8 * 2 * numRepeats
    mfolder_structure = ["ctmqc", 'model', 'mom']
    all_velMultiplier = [4, 3, 3, 2.5,
                         1, 1, 1.6, 1.5] * 2 * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000,
                   6000, 5000, 2500, 6000] * 2 * numRepeats
    all_model = [4, 3, 2, 1,
                 4, 3, 2, 1] * 2 * numRepeats
    all_p_mean = [-20, -15, -8, -20,
                  -20, -15, -8, -20] * 2 * numRepeats
    all_doCTMQC_C = (([True] * 8) + ([False] * 8)) * numRepeats
    all_doCTMQC_F = (([True] * 8) + ([False] * 8)) * numRepeats

    # Remember to change the smoothing choice when changing this!
    rootFolder = '%s/Test' % rootSaveFold

    print("Saving data in `%s'" % rootFolder)

elif inputs == "BigEhren":
    print("Carrying out full gossel simulations (ehrenfest and ctmqc for all momenta)")
    numRepeats = 1
    all_nRep = [50] * numRepeats
    mfolder_structure = ["ctmqc", 'model', 'mom']
    all_velMultiplier = [3] * numRepeats
    all_maxTime = [4000*3.8] * numRepeats
    all_model = ['big'] * numRepeats
    all_p_mean = [-20] * numRepeats
    all_doCTMQC_C = [False] * numRepeats
    all_doCTMQC_F = [False] * numRepeats

    # Remember to change the smoothing choice when changing this!
    #rootFolder = '%s/bigHam_100Reps_SmoothOnEh' % rootSaveFold
    rootFolder = '%s/bigHam_50Reps' % rootSaveFold

    print("Saving data in `%s'" % rootFolder)

elif inputs == "BigCTMQC":
    print("Carrying out full gossel simulations (ehrenfest and ctmqc for all momenta)")
    numRepeats = 1
    all_nRep = [100] * numRepeats
    mfolder_structure = ["ctmqc", 'model', 'mom']
    all_velMultiplier = [3] * numRepeats
    all_maxTime = [4000*4] * numRepeats
    all_model = ['big'] * numRepeats
    all_p_mean = [-20] * numRepeats
    all_doCTMQC_C = [True] * numRepeats
    all_doCTMQC_F = [True] * numRepeats

    # Remember to change the smoothing choice when changing this!
    rootFolder = '%s/bigHam_100Reps_SmoothOn' % rootSaveFold

    print("Saving data in `%s'" % rootFolder)

elif inputs == "BigBoth":
    print("Carrying out the multiple avoided crossings simulations (ehrenfest and ctmqc for all momenta)")
    numRepeats = 1
    all_nRep = [100, 100] * numRepeats
    mfolder_structure = ["ctmqc", 'model', 'mom']
    all_velMultiplier = [2, 2] * numRepeats
    all_maxTime = [70000, 70000] * numRepeats
    all_model = ['mult1', 'mult1'] * numRepeats
    all_p_mean = [-20, -20] * numRepeats
    all_doCTMQC_C = [True, False] * numRepeats
    all_doCTMQC_F = [True, False] * numRepeats

    # Remember to change the smoothing choice when changing this!
    rootFolder = '%s/multCrossings_100Reps_wRenorm' % rootSaveFold

    print("Saving data in `%s'" % rootFolder)

elif inputs == "constHighBoth":
    print("Carrying out the multiple avoided crossings simulations (ehrenfest and ctmqc for all momenta)")
    numRepeats = 1
    all_nRep = [100, 100] * numRepeats
    mfolder_structure = ["ctmqc", 'model', 'mom']
    all_velMultiplier = [2, 2] * numRepeats
    all_maxTime = [70000, 70000] * numRepeats
    all_model = ['constHigh', 'constHigh'] * numRepeats
    all_p_mean = [-20, -20] * numRepeats
    all_doCTMQC_C = [True, False] * numRepeats
    all_doCTMQC_F = [True, False] * numRepeats

    # Remember to change the smoothing choice when changing this!
    rootFolder = '%s/constHigh_100Reps_wRenorm' % rootSaveFold

    print("Saving data in `%s'" % rootFolder)

elif inputs == "PartGossel":
    print("Carrying out full gossel simulations (ehrenfest and ctmqc for all momenta)")
    numRepeats = 1
    all_nRep = [100] * 8 * 2 * numRepeats
    mfolder_structure = ["ctmqc", 'model', 'mom']
    all_velMultiplier = [4, 3,
                         1, 1.6] * 2 * numRepeats
    all_maxTime = [2000, 3500,
                   6000, 4500] * 2 * numRepeats
    all_model = [4, 2,
                 4, 2] * 2 * numRepeats
    all_p_mean = [-20, -13,
                  -20, -13] * 2 * numRepeats
    all_doCTMQC_C = (([True] * 4) + ([False] * 4)) * numRepeats
    all_doCTMQC_F = (([True] * 4) + ([False] * 4)) * numRepeats

    # Remember to change the smoothing choice when changing this!
    rootFolder = '%s/FullGossel_Rlk_100Reps_diabProp_part' % rootSaveFold

    print("Saving data in `%s'" % rootFolder)



elif inputs == "FullEhrenGossel":
    print("Carrying out Ehrenfest simulations!")
    numRepeats = 10
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = [4, 3, 3, 2.5,      1,    1,    1.6,  1.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000,  6000, 5000, 2500, 6000] * numRepeats
    all_model = [4, 3, 2, 1,                4,    3,    2,    1] * numRepeats
    all_p_mean = [-20, -15, -8, -20,        -20, -15,  -8,   -20] * numRepeats
    all_doCTMQC_C = ([False] * 8) * numRepeats
    all_doCTMQC_F = ([False] * 8 )  * numRepeats
    rootFolder = '%s/FullEhrenGossel' % rootSaveFold
    all_nRep = [200] * 8 * numRepeats

elif inputs == "FullCTMQCGossel":
    print("Carrying out full Gossel CTMQC testing!")
    numRepeats = 1
    mfolder_structure = ["ctmqc", "sigma", 'model', 'mom']
    all_velMultiplier = [4, 3, 3, 2.5,
                         1, 1, 1.6, 1.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000,
                   6000, 5000, 2500, 6000] * numRepeats
    all_model = [4, 3, 2, 1,
                 4, 3, 2, 1] * numRepeats
    all_p_mean = [-20, -15, -8, -20,
                  -20, -15, -8, -20] * numRepeats
    all_doCTMQC_C = ([True] * 8) * numRepeats
    all_doCTMQC_F = ([True] * 8)  * numRepeats
    rootFolder = '%s/CTMQC_Data/VaryingSigma' % rootSaveFold
    #all_dt = [0.4] * 8 * numRepeats
    all_nRep = [100] * 8 * numRepeats

elif inputs == "LowCTMQCGossel":
    print("Carrying out full Gossel CTMQC testing!")
    numRepeats = 3
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = [1, 1, 1.6, 1.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000] * numRepeats
    all_model = [4, 3, 2, 1] * numRepeats
    all_p_mean = [-20, -15, -8, -20] * numRepeats
    all_doCTMQC_C = ([True] * 4) * numRepeats
    all_doCTMQC_F = ([True] * 4)  * numRepeats
    rootFolder = '%s/CTMQC_Data/Test_RI0_Switch' % rootSaveFold
    all_dt = [0.4] * 4 * numRepeats
    all_nRep = [200] * 4 * numRepeats

elif inputs == "HighCTMQCGossel":
    print("Carrying out full Gossel CTMQC testing!")
    numRepeats = 1
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = [4, 3, 3, 2.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000] * numRepeats
    all_model = [4, 3, 2, 1] * numRepeats
    all_p_mean = [-20, -15, -8, -20] * numRepeats
    all_doCTMQC_C = ([True] * 4) * numRepeats
    all_doCTMQC_F = ([True] * 4)  * numRepeats
    rootFolder = '%s/CTMQC_Data/Adiab_Prop' % rootSaveFold
    #all_dt = [0.4] * 4 * numRepeats
    all_nRep = [100] * 4 * numRepeats

elif inputs == "HighVaryingSigma":
    print("Carrying out varying sigma CTMQC testing!")
    numRepeats = 8
    mfolder_structure = ['sigma', 'model', 'mom']
    all_velMultiplier = [4, 3, 3, 2.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000] * numRepeats
    all_model = [4, 3, 2, 1] * numRepeats
    all_p_mean = [-20, -15, -8, -20] * numRepeats
    all_doCTMQC_C = ([True] * 4) * numRepeats
    all_doCTMQC_F = ([True] * 4)  * numRepeats
    rootFolder = '%s/VaryingSigma' % rootSaveFold
    #all_dt = [0.4] * 4 * numRepeats
    all_nRep = [100] * 4 * numRepeats
    all_sig_min = [j for j in (0.1, 0.2, 0.3, 0.5, 0.6, 0.75, 1, 2) for i in range(4)]


elif inputs == "changingNumReps":
    print("Carrying out custom input file")
    numRepeats = 1  # How many repeated simulations (each with different init pos)
    mfolder_structure = ['nrep', 'model', 'mom']  # What the folderstructure of the outputted data looks like.
    all_nRep = [2, 5, 10, 35, 75, 150, 250, 500, 750, 1000] * numRepeats  # How many replicas
    numRepeats *= len(all_nRep)
    all_model = [2] * numRepeats # What tully model to use
    all_velMultiplier = [3] * numRepeats # What momentum to select (this is divided by 10 so type 3 for 30)
    all_maxTime = [1500] * numRepeats # How long to run for
    all_p_mean = [-15] * numRepeats # The average initial position
    all_doCTMQC_C = [True] * numRepeats # Whether to use the coeff CTMQC equations
    all_doCTMQC_F = [True]  * numRepeats # Whether the use the frc CTMQC equations
    rootFolder = './ChangingNumReps'  #'%s/test' % rootSaveFold  # Where to save the data.

else:
    print("Carrying out custom input file")
    numRepeats = 1#len(all_dt)  # How many repeated simulations (each with different init pos)
    mfolder_structure = ["ctmqc", 'model', 'mom']  # What the folderstructure of the outputted data looks like.
    all_nRep = [100] * numRepeats  # How many replicas
    all_model = [4] * numRepeats  # What tully model to use
    all_velMultiplier = [4] * numRepeats  # What momentum to select (this is divided by 10 so type 3 for 30)
    all_maxTime = [2000] * numRepeats  # How long to run for
    all_p_mean = [-20] * numRepeats  # The average initial position
    all_doCTMQC_C = [True] * numRepeats  # Whether to use the coeff CTMQC equations
    all_doCTMQC_F = [True]  * numRepeats  # Whether the use the frc CTMQC equations
    #all_dt = [0.4134137457575]
    rootFolder = './Test'  #'%s/test' % rootSaveFold  # Where to save the data.



import os
if os.path.isdir(rootFolder) and inputs != 'custom':
   print("The directory '%s' already exists, remove it to continue." % rootFolder)
   raise SystemExit("The directory '%s' already exists, remove it to continue." % rootFolder)



print("\n\nAll sim params:")
s = ' '.join([f"{i:8s}" for i in ("Nrep","Model","Vel","Pos","Sigma","CoefCT","FrcCT")])
print("" + '='*len(s) + "\n" + s +"\n" + '-'*len(s))
for i in range(min([len(i) for i in (all_nRep, all_model, all_velMultiplier, all_p_mean, all_sig_min, all_doCTMQC_C, all_doCTMQC_F)])):
    s = ""
    for j in (all_nRep, all_model, all_velMultiplier, all_p_mean, all_sig_min, all_doCTMQC_C, all_doCTMQC_F):
        s += f"{str(j[i]):8s} "
    print(s)
print("\n" + '='*len(s) + "\n")

