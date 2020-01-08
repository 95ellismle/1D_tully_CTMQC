inputs = "custom"
s_min = 0.5
mass = 2000 
rootSaveFold = "/scratch/mellis/TullyModelData/Test"
do_parallel = True



modelTime = {1: (6000, 4000), 2: (2500, 1500), 3: (5000, 1500), 4: (6000, 2000)}
modelVel = {1: (1.5, 2.5), 2: (1.6, 3), 3: (1, 3), 4: (1, 4)}
modelPos = {1: -20, 2: -8, 3: -15, 4: -20}


all_dt = False
all_elec_steps = False
if inputs == "FullGossel":
    print("Carrying out full gossel simulations (ehrenfest and ctmqc for all momenta)")
    numRepeats = 1
    all_nRep = [200] * 8 * 2 * numRepeats
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
    rootFolder = '%s/FullGossel_VarSigRI0' % rootSaveFold

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
    mfolder_structure = ["ctmqc", 'model', 'mom']
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
    rootFolder = '%s/CTMQC_Data/VaryingSigma20' % rootSaveFold
    all_nRep = [200] * 8 * numRepeats

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
    numRepeats = 3
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = [4, 3, 3, 2.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000] * numRepeats
    all_model = [4, 3, 2, 1] * numRepeats
    all_p_mean = [-20, -15, -8, -20] * numRepeats
    all_doCTMQC_C = ([True] * 4) * numRepeats
    all_doCTMQC_F = ([True] * 4)  * numRepeats
    rootFolder = '%s/CTMQC_Data/Adiab_Prop' % rootSaveFold
    all_dt = [0.4] * 4 * numRepeats
    all_nRep = [200] * 4 * numRepeats

elif 'norm' in inputs:
    import re
    models = [int(i) for i in re.findall("[0-9]", inputs)]
    if len(models) == 0:   models = [1, 2, 3, 4]
    
    all_elec_steps = [1, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000]
    models = models * len(all_elec_steps)
    numRepeats = 1


    mfolder_structure = ['model', 'mom']
    all_velMultiplier = [modelVel[i][1] for i in models] * numRepeats
    all_maxTime = [modelTime[i][1] for i in models] * numRepeats
    all_model = models * numRepeats
    all_p_mean = [modelPos[i] for i in models] * numRepeats
    all_doCTMQC_C = ([True] * len(models)) * numRepeats
    all_doCTMQC_F = ([True] * len(models))  * numRepeats
    rootFolder = '%s/CTMQC_Data/Adiab_Prop' % rootSaveFold
    all_dt = [0.4] * len(models) * numRepeats
    all_nRep = [70] * len(models) * numRepeats
 

else:

    print("Carrying out custom input file")
    numRepeats = 1  # How many repeated simulations (each with different init pos)
    #mfolder_structure = ['sigma', 'model', 'mom']  # What the folderstructure of the outputted data looks like.
    #all_nRep = [70] * numRepeats  # How many replicas
    #all_model = [3] * numRepeats  # What tully model to use
    #all_velMultiplier = [3] * numRepeats  # What momentum to select (this is divided by 10 so type 3 for 30)
    #all_maxTime = [1500] * numRepeats  # How long to run for
    #all_p_mean = [-15] * numRepeats  # The average initial position
    #all_doCTMQC_C = [True] * numRepeats  # Whether to use the coeff CTMQC equations
    #all_doCTMQC_F = [True]  * numRepeats  # Whether the use the frc CTMQC equations
    #rootFolder = './test'  #'%s/test' % rootSaveFold  # Where to save the data.
    mfolder_structure = ["ctmqc", 'model', 'mom']
    all_velMultiplier = [3,] * numRepeats
    all_maxTime = [1500,] * numRepeats
    all_model = [3,] * numRepeats
    all_p_mean = [-8,] * numRepeats
    all_doCTMQC_C = ([True] * 1) * numRepeats
    all_doCTMQC_F = ([True] * 1)  * numRepeats
    rootFolder = './test'
    all_nRep = [5] * 1 * numRepeats



import os
if os.path.isdir(rootFolder) and inputs != 'custom':
   print("The directory '%s' already exists, remove it to continue." % rootFolder)
   raise SystemExit("The directory '%s' already exists, remove it to continue." % rootFolder)
