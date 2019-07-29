#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:54:51 2019

@author: mellis
"""

"""
        ###   Now Plot the Data   ###
    """
    
    if data.ctmqc_env['iter'] < 30:
        whichPlot = ""
    
    # Actually do the plotting
    if isinstance(whichPlot, str):
        whichPlot = whichPlot.lower()
        if 'qm_fl_fk' in whichPlot:
            fig, axes = plt.subplots(2)
            # Plot the QM and adiabatic momentum
    
            ax = axes[0]
            fl_fk = (data.allAdMom[:, :, 0, 0] - data.allAdMom[:, :, 0, 1])
            QM = data.allQlk[:, :, 0, 0, 1] * data.ctmqc_env['mass']
            fl_fk *= QM
    
            def init_QM_fl_fk():
                minR, maxR = np.min(data.allR), np.max(data.allR)
                minflQ, maxflQ = np.min(fl_fk), np.max(fl_fk)
                minQ, maxQ = np.min(QM), np.max(QM)
    
                ln1, = ax.plot(np.linspace(minR, maxR, QM.shape[1]),
                               np.linspace(minflQ, maxflQ, QM.shape[1]), 'y.')
                ln2, = ax.plot(np.linspace(minR, maxR, QM.shape[1]),
                               np.linspace(minQ, maxQ, QM.shape[1]),
                               '.', color=(1, 0, 1))
                return ln1, ln2
    
            def ani_init_2():
                return ln1, ln2
    
            def animate_QM_fl_fk(i):
                step = i % len(data.allt)
                ln1.set_ydata(fl_fk[step, :])  # update the data.
                ln1.set_xdata(data.allR[step, :, 0])
                ln2.set_ydata(QM[step, :])  # update the data.
                ln2.set_xdata(data.allR[step, :, 0])
                return ln1, ln2
    
            ln1, ln2 = init_QM_fl_fk()
            axes[1].plot(data.allR[:, :, 0], data.allE[:, 0, 0])
    
            ani = animation.FuncAnimation(
                    fig, animate_QM_fl_fk, init_func=ani_init_2,
                    interval=10, blit=True)
    
            plt.show()
    
        # Nuclear Density
        if whichPlot == 'nucl_dens':
            nstep, nrep, natom = np.shape(data.allR)
            allND = [qUt.calc_nucl_dens_PP(R, sig)
                     for R, sig in zip(data.allR, data.allSigma)]
            allND = np.array(allND)
            minR, maxR = np.min(data.allR), np.max(data.allR)
            minND, maxND = np.min(allND[:, 0]), np.max(allND[:, 0])
            f, axes = plt.subplots(1)
    
            ln, = axes.plot(np.linspace(minR, maxR, nrep),
                            np.linspace(minND, maxND, nrep),
                            '.', color=(1., 0., 1.))
    
            def ani_init_1():
                for I in range(nrep):
                    axes.plot(data.allR[:, I, 0], data.allE[:, I, 0],
                              lw=0.3, alpha=0.2)
                return ln,
    
            def animate_ND(i):
                step = i % nstep
                ln.set_ydata(allND[step][0])
                ln.set_xdata(allND[step][1])
                return ln,
    
            ani = animation.FuncAnimation(
                    f, animate_ND, init_func=ani_init_1, interval=10, blit=True)
    
            plt.show()
    
        R = data.allR[:, 0]
        if '|c|^2' in whichPlot:
            plt.figure()
            # Plot ad coeffs
            for I in range(data.ctmqc_env['nrep']):
                params = {'lw': 0.5, 'alpha': 0.1, 'color': 'r'}
                plot.plot_ad_pops(data.allt, data.allAdPop[:, I, 0, 0], params)
                params = {'lw': 0.5, 'alpha': 0.1, 'color': 'b'}
                plot.plot_ad_pops(data.allt, data.allAdPop[:, I, 0, 1], params)
    
            avgData = np.mean(data.allAdPop, axis=1)
            params = {'lw': 2, 'alpha': 1, 'ls': '--', 'color': 'r'}
            plot.plot_ad_pops(data.allt, avgData[:, 0, 0], params)
            params = {'lw': 2, 'alpha': 1, 'ls': '--', 'color': 'b'}
            plot.plot_ad_pops(data.allt, avgData[:, 0, 1], params)
            plt.xlabel("Time [au_t]")
            plt.annotate(r"K$_0$ = %.1f au" % (v_mean * data.ctmqc_env['mass']),
                         (10, 0.5), fontsize=24)
    
    #        plt.title("Sigma = %.2f" % s_mean)
    #        plt.savefig("/home/oem/Documents/PhD/Graphs/1D_Tully/Model1/CTMQC_25K/ChangingSigma/%.2f_pops.png" % s_mean)
    
        if 'deco' in whichPlot:
            # Plot Decoherence
            plt.figure()
            allDeco = data.allAdPop[:, :, :, 0] * data.allAdPop[:, :, :, 1]
            avgDeco = np.mean(allDeco, axis=1)
            plt.plot(data.allt, avgDeco)
    
            minD, maxD = np.min(avgDeco), np.max(avgDeco)
            rD = maxD - minD
            plt.annotate(r"K$_0$ = %.1f au" % (v_mean * data.ctmqc_env['mass']),
                         (10, minD+(rD/2.)), fontsize=24)
            plt.ylabel("Coherence")
            plt.xlabel("Time [au_t]")
    
    #        plt.title("Sigma = %.2f" % s_mean)
    #        plt.savefig("/home/oem/Documents/PhD/Graphs/1D_Tully/Model1/CTMQC_25K/ChangingSigma/%.2f_deco.png" % s_mean)
    
        if 'adFrc' in whichPlot:
            dx = data.allv[:, 0, 0] * data.ctmqc_env['dt']
            
    
        if 'rlk' in whichPlot:
            f, a = plt.subplots()
            ln_lk = a.plot(data.allt, data.allRlk[:, 0, 0, 1], 'b-', lw=1, alpha=1)
            ln_I0 = a.plot(data.allt, data.allRI0[:, :, 0],
                           'k--', lw=0.7, alpha=0.5)
            
            avgRI0 = np.average(data.allRI0[:, :, 0], axis=1)
            stdRI0 = np.std(data.allRI0[:, :, 0], axis=1)
            minRI0 = np.min(data.allRI0[:, :, 0], axis=1)
            maxRI0 = np.max(data.allRI0[:, :, 0], axis=1)
    #        a.plot(data.allt, avgRI0, 'k--')
            a.plot(data.allt, maxRI0 + 2*stdRI0, 'g--')
            a.plot(data.allt, minRI0 - 2*stdRI0, 'g--')
            
            a.set_xlabel("Time [au_t]")
            a.set_ylabel(r"Rlk [bohr$^{-1}$]")
            a.legend([ln_lk[0], ln_I0[0]], [r'$\mathbf{R}_{lk, \nu}^{0}$',
                                            r"$\mathbf{R}_{0, \nu}^{I}$"])
            
    
        if '|u|^2' in whichPlot:
            plt.figure()
            plot.plot_di_pops(data.allt, data.allu, "Time")
        if 'norm' in whichPlot:
            norm = np.linalg.norm(data.allC, axis=3)
            plt.figure()
            plt.plot(data.allt, norm[:, :, 0], lw=0.7, alpha=0.5)
            plt.plot(data.allt, np.mean(norm[:, :, 0], axis=1), lw=1.3, ls='--')
            plt.xlabel("Time [au_t]")
            plt.ylabel("Norm (adPops)")
        if 'rabi' in whichPlot:
            plot.plot_Rabi(data.allt, data.allH[0, 0])
        if 'ham' in whichPlot:
            plot.plot_H(data.allt, data.allH, "Time")
    
        if 'ham_ax' in whichPlot:
            plot.plot_H_all_x(data.ctmqc_env)
    
        if 'eh_frc_ax' in whichPlot:
            for i in np.arange(0, 1, 0.1):
                pops = [1-i, i]
                data.ctmqc_env['C'] = np.array([[complex(np.sqrt(1-i), 0),
                                                 complex(np.sqrt(i), 0)]])
                plot.plot_eh_frc_all_x(data.ctmqc_env, label=r"$|C|^2 = $%.1g" % i)
        if 'ad_frc_ax' in whichPlot:
            plot.plot_adFrc_all_x(data.ctmqc_env)
        if 'ad_ener_ax' in whichPlot:
            plot.plot_ener_all_x(data.ctmqc_env)
        if 'nacv_ax' in whichPlot:
            plot.plot_NACV_all_x(data.ctmqc_env)
    
        plt.show()
    #    plt.close("all")