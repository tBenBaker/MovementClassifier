import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


#note aist indexes = 0nose, 1L-ear, 2R-ear, 3L-shoulder, 4R-shoulder, 5L-elbow, 6R-elbow, 7L-wrist, 
# 8R-wrist, 9L-hip, 10R-hip, 11L-knee, 12R-knee, 13L-ankle, 14R-ankle 


class Dance:
    
    def __init__(self, pos, dt):        #pos is a 3d array of joint positions, dt is the time between frames
        self.pos = pos
        self.numjoints = len(self.pos)
        self.dt = dt
        self.frames = range(len(pos[0]))
        self.numframes = len(self.frames)
        self.times = [dt * f for f in self.frames]
        self.moment = 1/dt * 1/4
        self.moments = np.arange(self.moment, self.numframes, self.moment, dtype=int) 
        #quarter-second intervals across frames, for measuring feature stats across moments
        
        self.id = []        #initialize variables to be set below
        self.genre = []       
        self.velocity = []
        self.acceleration = []
        self.jerk = []
        self.movedata = []
        self.sacrum = []
        
        
        self.features = {} #initialize dictionary of features

    def get_movedata(self):
        #calculate deriv, smooth, repeat 3x for velocity, acceleration, jerk

        rawvel = (self.pos[:, 1:] - self.pos[:, :-1]) / self.dt
        vel = np.empty_like(rawvel)                    
        
        for index, joint in enumerate(rawvel):
            for dim in range(3):                        #savgol smoothing filter. window empirically set 
                vel[index, ..., dim] = savgol_filter(joint[:,dim], window_length=45, polyorder=2, mode='nearest')  
        
        vel = np.pad(vel, ((0,0), (0,1), (0,0)), mode='edge') #pad last frame to maintain sizing                                   
        
        rawacc = (vel[:, 1:] - vel[:, :-1]) / self.dt       #derive acceleration
        acc = np.empty_like(rawacc)
        
        for index, joint in enumerate(rawacc):         #smoothing filter
            for dim in range(3):                        
                acc[index, ..., dim] = savgol_filter(joint[:,dim], window_length=45, polyorder=2, mode='nearest')
        acc = np.pad(acc, ((0,0), (0,1), (0,0)), mode='edge')
        
        rawjerk = (acc[:, 1:] - acc[:, :-1]) / self.dt      #derive jerk
        jerk = np.empty_like(rawjerk)
        
        for index, joint in enumerate(rawjerk):         #smoothing filter
            for dim in range(3):                        
                jerk[index, ..., dim] = savgol_filter(joint[:,dim], window_length=45, polyorder=2, mode='nearest')
        jerk = np.pad(jerk, ((0,0), (0,1), (0,0)), mode='edge')
        
        self.velocity = vel
        self.acceleration = acc
        self.jerk = jerk
        self.movedata = [self.pos, self.velocity, self.acceleration, self.jerk]    

    def get_sacrum(self, hipidxs=[9,10]):       #populate virtual sacrum joint by averaging hip positions.   
                                                                    #idxs of AIST are [9,10]
        Lhip, Rhip = hipidxs
        sacrumpos = np.empty_like(self.pos[0])
        
        for f in range(self.numframes):
            sacrumpos[f] = (self.pos[Lhip][f] + self.pos[Rhip][f]) / 2     
        
        sacrumvel = (sacrumpos[1:, :] - sacrumpos[:-1, :]) / self.dt
        for dim in range(3):                        
            sacrumvel[:, dim] = savgol_filter(sacrumvel[:,dim], window_length=45, polyorder=2, mode='nearest')  
        sacrumvel = np.pad(sacrumvel, ((0,1), (0,0)), mode='edge')   

        sacrumacc = (sacrumvel[1:,:] - sacrumvel[:-1,:]) / self.dt
        for dim in range(3):                        
            sacrumacc[:, dim] = savgol_filter(sacrumacc[:,dim], window_length=45, polyorder=2, mode='nearest')  
        sacrumacc = np.pad(sacrumacc, ((0,1), (0,0)), mode='edge')  

        sacrumjer = (sacrumacc[1:,:] - sacrumacc[:-1,:]) / self.dt
        for dim in range(3):                        
            sacrumjer[:, dim] = savgol_filter(sacrumjer[:,dim], window_length=45, polyorder=2, mode='nearest')  
        sacrumjer = np.pad(sacrumjer, ((0,1), (0,0)), mode='edge')
        
        self.sacrum = [sacrumpos, sacrumvel, sacrumacc, sacrumjer] 
        
    def get_expandedness(self):   #expandedness ~ distance of joints from sacrum
        
        self.get_sacrum()
        Dsfromsacrum = np.empty_like(self.pos)  #inits for diff in pos, vel, and accel from sacrum
        Vsfromsacrum = np.empty_like(self.pos)  
        Asfromsacrum = np.empty_like(self.pos)  
                
        for j in range(self.numjoints):                         #distance of each joint from sacrum                                    
            Dsfromsacrum[j] = np.abs((self.pos[j] - self.sacrum[0]))
        
        expa = Dsfromsacrum.sum(axis=0)                         #sum over joints to get expandedness per frame per dimension
        expa = expa.sum(axis=1)                                 #sum over dimensions to get expandedness over frames
        meanexpa = expa.sum()/self.numframes                    #mean expandedness over frames
        stdexpa = expa.std()                                    #std of expandedness over frames
        
        for j in range(self.numjoints):                         #diff in velocity of each joint from sacrum
            Vsfromsacrum[j] = np.abs((self.velocity[j] - self.sacrum[1]))
        
        expavel = Vsfromsacrum.sum(axis=0)                      #velocity of expansion/compression per frame per dimension
        expavel = expavel.sum(axis=1)                           #velocity of expansion/compression over frames
        meanexpavel = expavel.sum()/self.numframes          
        stdexpavel = expavel.std()

        for j in range(self.numjoints):                         #diff in acceleration of each joint from sacrum      
            Asfromsacrum[j] = np.abs((self.acceleration[j] - self.sacrum[2]))

        expaacc = Asfromsacrum.sum(axis=0)                      #acceleration of expansion/compression per frame per dimension
        expaacc = expaacc.sum(axis=1)                           #acceleration of expansion/compression over frames
        meanexpaacc = expaacc.sum()/self.numframes
        stdexpaacc = expaacc.std()      
        
        self.features['Expandedness'] = meanexpa
        self.features['Expandedness_std'] = stdexpa
        self.features['Expandednessvel'] = meanexpavel
        self.features['Expandednessvel_std'] = stdexpavel
        self.features['Expandednessacc'] = meanexpaacc
        self.features['Expandednessacc_std'] = stdexpaacc
        
              
    def get_asymmetries(self, Ridxs=[4,6,8,10,12,14], Lidxs=[3,5,7,9,11,13], 
        Inidxs=[3, 4, 9, 10], Outidxs=[7, 8, 13, 14],                          #asymmetries ~ difference in joint positions
        Topidxs=[3, 4, 5, 6, 7, 8], Botidxs=[9, 10, 11, 12, 13, 14]):           #default values for aist++ dataset
                        
                                                                                             
        Rvel, Lvel, Racc, Lacc, Rjer, Ljer = [np.zeros(self.numframes) for i in range(6)]
        
        for j in Ridxs:
            Rvel += np.sum(self.velocity[j], axis=1)            #sum joints over dimensions on each side
            Racc += np.sum(self.acceleration[j], axis=1)
            Rjer += np.sum(self.jerk[j], axis=1)
            
        for j in Lidxs:
            Lvel += np.sum(self.velocity[j], axis=1)
            Lacc += np.sum(self.acceleration[j], axis=1)
            Ljer += np.sum(self.jerk[j], axis=1)

        velratio = Rvel / Lvel
        accelratio = Racc / Lacc
        jerkratio = Rjer / Ljer
              
        self.features['Asym_RL_vel'] = np.sum(velratio)   
        self.features['Asym_RL_acc'] = np.sum(accelratio) 
        self.features['Asym_RL_jer'] = np.sum(jerkratio) 
        
        velratiomoments = np.split(velratio, self.moments)             #split each div by moment = 15frames = 1/4sec
        accelratiomoments = np.split(accelratio, self.moments)
        jerkratiomoments = np.split(jerkratio, self.moments)
        
        velmeans = [[] for i in range(len(self.moments))]
        accelmeans = [[] for i in range(len(self.moments))]
        jerkmeans = [[] for i in range(len(self.moments))]
        
        for m in range(len(self.moments)):                              #take mean of each moment
            velmeans[m] = np.mean(velratiomoments[m])
            accelmeans[m] = np.mean(accelratiomoments[m])
            jerkmeans[m] = np.mean(jerkratiomoments[m])
            
        self.features['Asym_RL_vel_std'] = np.std(velmeans)
        self.features['Asym_RL_acc_std'] = np.std(accelmeans)
        self.features['Asym_RL_jer_std'] = np.std(jerkmeans)
        
        #repeat above for inside vs outside asymmetry 
    
        Invel, Outvel, Inacc, Outacc, Injer, Outjer = [np.zeros(self.numframes) for i in range(6)]
        
        for j in Inidxs:
            Invel += np.sum(self.velocity[j], axis=1)            
            Inacc += np.sum(self.acceleration[j], axis=1)
            Injer += np.sum(self.jerk[j], axis=1)
            
        for j in Outidxs:
            Outvel += np.sum(self.velocity[j], axis=1)
            Outacc += np.sum(self.acceleration[j], axis=1)
            Outjer += np.sum(self.jerk[j], axis=1)

        velratio = Invel / Outvel
        accelratio = Inacc / Outacc
        jerkratio = Injer / Outjer
              
        self.features['Asym_IO_vel'] = np.sum(velratio)   
        self.features['Asym_IO_acc'] = np.sum(accelratio) 
        self.features['Asym_IO_jer'] = np.sum(jerkratio) 
        
        velratiomoments = np.split(velratio, self.moments)             
        accelratiomoments = np.split(accelratio, self.moments)
        jerkratiomoments = np.split(jerkratio, self.moments)
        
        velmeans = [[] for i in range(len(self.moments))]
        accelmeans = [[] for i in range(len(self.moments))]
        jerkmeans = [[] for i in range(len(self.moments))]
        
        for m in range(len(self.moments)):                              
            velmeans[m] = np.mean(velratiomoments[m])
            accelmeans[m] = np.mean(accelratiomoments[m])
            jerkmeans[m] = np.mean(jerkratiomoments[m])
            
        self.features['Asym_IO_vel_std'] = np.std(velmeans)
        self.features['Asym_IO_acc_std'] = np.std(accelmeans)
        self.features['Asym_IO_jer_std'] = np.std(jerkmeans)
        
        #repeat above for top vs bottom asymmetry
        
        Topvel, Botvel, Topacc, Botacc, Topjer, Botjer = [np.zeros(self.numframes) for i in range(6)]
        
        for j in Topidxs:
            Topvel += np.sum(self.velocity[j], axis=1)            
            Topacc += np.sum(self.acceleration[j], axis=1)
            Topjer += np.sum(self.jerk[j], axis=1)
            
        for j in Outidxs:
            Botvel += np.sum(self.velocity[j], axis=1)
            Botacc += np.sum(self.acceleration[j], axis=1)
            Botjer += np.sum(self.jerk[j], axis=1)

        velratio = Topvel / Botvel
        accelratio = Topacc / Botacc
        jerkratio = Topjer / Botjer
              
        self.features['Asym_TB_vel'] = np.sum(velratio)   
        self.features['Asym_TB_acc'] = np.sum(accelratio) 
        self.features['Asym_TB_jer'] = np.sum(jerkratio) 
        
        velratiomoments = np.split(velratio, self.moments)             
        accelratiomoments = np.split(accelratio, self.moments)
        jerkratiomoments = np.split(jerkratio, self.moments)
        
        velmeans = [[] for i in range(len(self.moments))]
        accelmeans = [[] for i in range(len(self.moments))]
        jerkmeans = [[] for i in range(len(self.moments))]
        
        for m in range(len(self.moments)):                              
            velmeans[m] = np.mean(velratiomoments[m])
            accelmeans[m] = np.mean(accelratiomoments[m])
            jerkmeans[m] = np.mean(jerkratiomoments[m])
            
        self.features['Asym_TB_vel_std'] = np.std(velmeans)
        self.features['Asym_TB_acc_std'] = np.std(accelmeans)
        self.features['Asym_TB_jer_std'] = np.std(jerkmeans) 

    def get_autocorr_features(self, npeaks=2):
        
        self.get_sacrum()
        
        #autocorr features for sacrum in up/down direction
        sacrum_ay_corr = np.correlate(self.sacrum[2][:,1], self.sacrum[2][:,1], mode='full')    
        peaks_ay = find_peaks(sacrum_ay_corr, height=0, prominence=1e-4)                   #find peaks in autocorrelation
        start = len(peaks_ay[0])//2                                                #find peak closest to zero lag
        heights_ay = peaks_ay[1]['peak_heights'][start:]                 #get heights of peaks
        prominences_ay = peaks_ay[1]['prominences'][start:]         #get prominences of peaks
                                      
        self.features['Autocorr_sac_height_ay'] = heights_ay[0]     #first peak height
        
        top_prom_ay = []                                    #prominences of top n peaks
        shortestpeak_ay = np.sort(heights_ay)[::-1][npeaks]    #height of nth tallest peak = shortest we care about
        for h in range(len(heights_ay)):            
            if heights_ay[h] >= shortestpeak_ay:
                top_prom_ay.append(prominences_ay[h])       #loop through peaks and get prominences of top n peaks 
                
        self.features['Autocorr_sac_prominence1_ay'] = np.sort(top_prom_ay)[::-1][0]
        self.features['Autocorr_sac_prominence2_ay'] = np.sort(top_prom_ay)[::-1][1]
        self.features['Autocorr_sac_prominence3_ay'] = np.sort(top_prom_ay)[::-1][2]
        #self.features['Autocorr_sac_prominence4_ay'] = np.sort(top_prom_ay)[::-1][3]
        self.features['Autocorr_prominences_ay_var'] = np.var(np.sort(top_prom_ay)[::-1])       #variance of prominences of top n peaks
        
        #autocorr features for sacrum in horizontal plane
        sacrum_axz_corr = np.correlate(self.sacrum[2][:,0], self.sacrum[2][:,0], mode='full') + np.correlate(self.sacrum[2][:,2], self.sacrum[2][:,2], mode='full')
        peaks_axz = find_peaks(sacrum_axz_corr, height=0, prominence=1e-4)
        start = len(peaks_axz[0])//2
        heights_axz = peaks_axz[1]['peak_heights'][start:]
        prominences_axz = peaks_axz[1]['prominences'][start:]    
        
        self.features['Autocorr_sac_height_axz'] = heights_axz[0]
        
        top_prom_axz = []
        shortestpeak_axz = np.sort(heights_axz)[::-1][npeaks]
        for h in range(len(heights_axz)):
            if heights_axz[h] >= shortestpeak_axz:
                top_prom_axz.append(prominences_axz[h])
                
        self.features['Autocorr_sac_prominence1_axz'] = np.sort(top_prom_axz)[::-1][0]
        self.features['Autocorr_sac_prominence2_axz'] = np.sort(top_prom_axz)[::-1][1]
        self.features['Autocorr_sac_prominence3_axz'] = np.sort(top_prom_axz)[::-1][2]
        #self.features['Autocorr_sac_prominence4_axz'] = np.sort(top_prom_axz)[::-1][3]
        self.features['Autocorr_sac_prominences_axz_var'] = np.var(np.sort(top_prom_axz)[::-1])
   
        #autocorr of combined wrists [7,8], combined ankles [13,14], per dimension 

        wristcorrx = (np.correlate(self.acceleration[7][:,0], self.acceleration[7][:,0], mode='full') + 
             np.correlate(self.acceleration[8][:,0], self.acceleration[8][:,0], mode='full')) / 2

        wristcorrz = (np.correlate(self.acceleration[7][:,2], self.acceleration[7][:,2], mode='full') + 
             np.correlate(self.acceleration[8][:,2], self.acceleration[8][:,2], mode='full')) / 2

        anklecorrx = (np.correlate(self.acceleration[13][:,0], self.acceleration[13][:,0], mode='full') + 
             np.correlate(self.acceleration[14][:,0], self.acceleration[14][:,0], mode='full')) / 2

        anklecorrz = (np.correlate(self.acceleration[13][:,2], self.acceleration[13][:,2], mode='full') + 
             np.correlate(self.acceleration[14][:,2], self.acceleration[14][:,2], mode='full')) / 2

        wristcorry = (np.correlate(self.acceleration[7][:,1], self.acceleration[7][:,1], mode='full') + 
             np.correlate(self.acceleration[8][:,1], self.acceleration[8][:,1], mode='full')) / 2

        anklecorry = (np.correlate(self.acceleration[13][:,1], self.acceleration[13][:,1], mode='full') + 
             np.correlate(self.acceleration[14][:,1], self.acceleration[14][:,1], mode='full')) / 2
            
        #autocorr features for wrists in horizontal plane
        
        peaks_wristsxz = find_peaks(wristcorrx + wristcorrz, height=0, prominence=1e-4)
        start = len(peaks_wristsxz[0])//2
        heights_wristsxz = peaks_wristsxz[1]['peak_heights'][start:]
        prominences_wristsxz = peaks_wristsxz[1]['prominences'][start:]    
                                      
        self.features['Autocorr_wrists_height_axz'] = heights_wristsxz[0]
        
        top_prom_wristsxz = []
        shortestpeak_wristsxz = np.sort(heights_wristsxz)[::-1][npeaks]
        for h in range(len(heights_wristsxz)):
            if heights_wristsxz[h] >= shortestpeak_wristsxz:
                top_prom_wristsxz.append(prominences_wristsxz[h])
                
        self.features['Autocorr_wrists_prominenceaxz1'] = np.sort(top_prom_wristsxz)[::-1][0]
        self.features['Autocorr_wrists_prominenceaxz2'] = np.sort(top_prom_wristsxz)[::-1][1]
        self.features['Autocorr_wrists_prominenceaxz3'] = np.sort(top_prom_wristsxz)[::-1][2]
        #self.features['Autocorr_wrists_prominence4'] = np.sort(top_prom_wristsxz)[::-1][3]
        self.features['Autocorr_wrists_prominenceaxz_std'] = np.std(np.sort(top_prom_wristsxz)[::-1])

        #autocorr features for wrists in up/down direction

        peaks_wristsy = find_peaks(wristcorry, height=0, prominence=1e-4)
        start = len(peaks_wristsy[0])//2
        heights_wristsy = peaks_wristsy[1]['peak_heights'][start:]
        prominences_wristsy = peaks_wristsy[1]['prominences'][start:]   

        self.features['Autocorr_wrists_height_ay'] = heights_wristsy[0]

        top_prom_wristsy = []
        shortestpeak_wristsy = np.sort(heights_wristsy)[::-1][npeaks]
        for h in range(len(heights_wristsy)):
            if heights_wristsy[h] >= shortestpeak_wristsy:
                top_prom_wristsy.append(prominences_wristsy[h])

        self.features['Autocorr_wrists_prominenceay1'] = np.sort(top_prom_wristsy)[::-1][0]
        self.features['Autocorr_wrists_prominenceay2'] = np.sort(top_prom_wristsy)[::-1][1]
        self.features['Autocorr_wrists_prominenceay3'] = np.sort(top_prom_wristsy)[::-1][2]
        #self.features['Autocorr_wrists_prominence4'] = np.sort(top_prom_wristsy)[::-1][3]
        self.features['Autocorr_wrists_prominenceay_std'] = np.std(np.sort(top_prom_wristsy)[::-1])

        #autocorr features for ankles in horizontal plane

        peaks_anklxz = find_peaks(anklecorrx + anklecorrz, height=0, prominence=1e-4)
        start = len(peaks_anklxz[0])//2
        heights_anklxz = peaks_anklxz[1]['peak_heights'][start:]  
        prominences_anklxz = peaks_anklxz[1]['prominences'][start:] 
                                      
        self.features['Autocorr_ankles_height_axz'] = heights_anklxz[0]
        
        top_prom_anklxz = []
        shortestpeak_anklxz = np.sort(heights_anklxz)[::-1][npeaks]
        for h in range(len(heights_anklxz)):
            if heights_anklxz[h] >= shortestpeak_anklxz:
                top_prom_anklxz.append(prominences_anklxz[h])
                
        self.features['Autocorr_ankle_prominenceaxz1'] = np.sort(top_prom_anklxz)[::-1][0] 
        self.features['Autocorr_ankle_prominence2axz'] = np.sort(top_prom_anklxz)[::-1][1]
        self.features['Autocorr_ankle_prominence3axz'] = np.sort(top_prom_anklxz)[::-1][2]
        #self.features['Autocorr_ankle_prominence4'] = np.sort(top_prom_anklxz)[::-1][3]
        self.features['Autocorr_ankles_prominenceaxz_std'] = np.std(np.sort(top_prom_anklxz)[::-1])

        #autocorr features for ankles in up/down direction

        peaks_ankly = find_peaks(anklecorry, height=0, prominence=1e-4)
        start = len(peaks_ankly[0])//2
        heights_ankly = peaks_ankly[1]['peak_heights'][start:]
        prominences_ankly = peaks_ankly[1]['prominences'][start:]    

        self.features['Autocorr_ankles_height_ay'] = heights_ankly[0]

        top_prom_ankly = []
        shortestpeak_ankly = np.sort(heights_ankly)[::-1][npeaks]
        for h in range(len(heights_ankly)):
            if heights_ankly[h] >= shortestpeak_ankly:
                top_prom_ankly.append(prominences_ankly[h])

        self.features['Autocorr_ankles_prominenceay1'] = np.sort(top_prom_ankly)[::-1][0]
        self.features['Autocorr_ankles_prominenceay2'] = np.sort(top_prom_ankly)[::-1][1]
        self.features['Autocorr_ankles_prominenceay3'] = np.sort(top_prom_ankly)[::-1][2]
        #self.features['Autocorr_ankles_prominence4'] = np.sort(top_prom_ankly)[::-1][3]
        self.features['Autocorr_ankles_prominenceay_std'] = np.std(np.sort(top_prom_ankly)[::-1])

        #corr for contralateral elbows and knees

        Relb_Lknee_contracorr = (np.correlate(self.acceleration[6][:,0], self.acceleration[11][:,0], mode='full') +
                                    np.correlate(self.acceleration[6][:,1], self.acceleration[11][:,1], mode='full') +
                                    np.correlate(self.acceleration[6][:,2], self.acceleration[11][:,2], mode='full'))/3

        peaks_contracorr_Relb_Lknee = find_peaks(Relb_Lknee_contracorr, height=0, prominence=1e-4)
        start = len(peaks_contracorr_Relb_Lknee[0])//2
        heights_contracorr_Relb_Lknee = peaks_contracorr_Relb_Lknee[1]['peak_heights'][start:]
        prominences_contracorr_Relb_Lknee = peaks_contracorr_Relb_Lknee[1]['prominences'][start:]

        self.features['Contracorr_Relb_Lknee_height'] = heights_contracorr_Relb_Lknee[0]

        top_prom_contracorr_Relb_Lknee = []
        shortestpeak_contracorr_Relb_Lknee = np.sort(heights_contracorr_Relb_Lknee)[::-1][npeaks]
        for h in range(len(heights_contracorr_Relb_Lknee)):
            if heights_contracorr_Relb_Lknee[h] >= shortestpeak_contracorr_Relb_Lknee:
                top_prom_contracorr_Relb_Lknee.append(prominences_contracorr_Relb_Lknee[h])

        self.features['Contracorr_Relb_Lknee_prominence1'] = np.sort(top_prom_contracorr_Relb_Lknee)[::-1][0]
        self.features['Contracorr_Relb_Lknee_prominence2'] = np.sort(top_prom_contracorr_Relb_Lknee)[::-1][1]
        self.features['Contracorr_Relb_Lknee_prominence3'] = np.sort(top_prom_contracorr_Relb_Lknee)[::-1][2]
        #self.features['Contracorr_Relb_Lknee_prominence4'] = np.sort(top_prom_contracorr_Relb_Lknee)[::-1][3]
        self.features['Contracorr_Relb_Lknee_prominence_std'] = np.std(np.sort(top_prom_contracorr_Relb_Lknee)[::-1])
        
        Lelb_Rknee_contracorr = (np.correlate(self.acceleration[5][:,0], self.acceleration[12][:,0], mode='full') +
                                    np.correlate(self.acceleration[5][:,1], self.acceleration[12][:,1], mode='full') +
                                    np.correlate(self.acceleration[5][:,2], self.acceleration[12][:,2], mode='full'))/3 

        peaks_contracorr_Lelb_Rknee = find_peaks(Lelb_Rknee_contracorr, height=0, prominence=1e-4)
        start = len(peaks_contracorr_Lelb_Rknee[0])//2
        heights_contracorr_Lelb_Rknee = peaks_contracorr_Lelb_Rknee[1]['peak_heights'][start:]
        prominences_contracorr_Lelb_Rknee = peaks_contracorr_Lelb_Rknee[1]['prominences'][start:]   

        self.features['Contracorr_Lelb_Rknee_height'] = heights_contracorr_Lelb_Rknee[0]

        top_prom_contracorr_Lelb_Rknee = []
        shortestpeak_contracorr_Lelb_Rknee = np.sort(heights_contracorr_Lelb_Rknee)[::-1][npeaks]
        for h in range(len(heights_contracorr_Lelb_Rknee)):
            if heights_contracorr_Lelb_Rknee[h] >= shortestpeak_contracorr_Lelb_Rknee:
                top_prom_contracorr_Lelb_Rknee.append(prominences_contracorr_Lelb_Rknee[h]) 

        self.features['Contracorr_Lelb_Rknee_prominence1'] = np.sort(top_prom_contracorr_Lelb_Rknee)[::-1][0]
        self.features['Contracorr_Lelb_Rknee_prominence2'] = np.sort(top_prom_contracorr_Lelb_Rknee)[::-1][1]
        self.features['Contracorr_Lelb_Rknee_prominence3'] = np.sort(top_prom_contracorr_Lelb_Rknee)[::-1][2]
        #self.features['Contracorr_Lelb_Rknee_prominence4'] = np.sort(top_prom_contracorr_Lelb_Rknee)[::-1][3]
        self.features['Contracorr_Lelb_Rknee_prominence_std'] = np.std(np.sort(top_prom_contracorr_Lelb_Rknee)[::-1])
        
    def get_features(self):
        
        self.get_movedata()
        self.get_expandedness()
        self.get_asymmetries()
        self.get_autocorr_features()
        self.features['id'] = self.id
        self.features['Genre'] = self.genre