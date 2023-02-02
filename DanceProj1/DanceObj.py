import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths
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
        
    def get_mofeatures(self, sparse=False):
        #angular momentum of each joint, summed over all joints
        #sparse = True returns minimal array of features 
        
        self.get_sacrum()
        angmom = np.empty_like(self.pos[0])
        
        for j in range(self.numjoints):
                angmom = np.cross(self.pos[j] - self.sacrum[0], np.abs(self.velocity[j]))
        angmomm = angmom.mean()
        
        if sparse==True:
            self.features['angularmomentum'] = angmomm
            self.features['angularmomentumstd'] = angmomm.std()

        #wrist and ankle acceleration
        Lwrist, Rwrist = [7,8] 
        Lankle, Rankle = [13,14]
       
        wristacc = (np.abs(self.acceleration[Lwrist]) + np.abs(self.acceleration[Rwrist])) / 2   
        ankleacc = (np.abs(self.acceleration[Lankle]) + np.abs(self.acceleration[Rankle])) / 2  

        self.features['wristacceleration'] = wristacc.mean()
        self.features['wristaccstd'] = wristacc.std()           
        self.features['ankleacceleration'] = ankleacc.mean()
        self.features['ankleaccstd'] = ankleacc.std()
        
        if sparse==False:

            self.features['angularmomentumxz'] = angmom[:,0].mean() + angmom[:,2].mean()
            self.features['angularmomentumy'] = angmom[:,1].mean()
            self.features['angularmomentumxzstd'] = angmom[:,0].std() + angmom[:,2].std()
            self.features['angularmomentumystd'] = angmom[:,1].std()
            self.features['angularmomentumnose'] = angmom[0].mean()
            self.features['angularmomentumnosestd'] = angmom[0].std()
            self.features['angularmomentumelbows'] = angmom[5].mean() + angmom[6].mean()
            self.features['angularmomentumelbowsstd'] = angmom[5].std() + angmom[6].std()

            ypeaks, yproperties = find_peaks(angmom[:,1], height=0, distance=30, prominence=500, width=10)
            yapeaks, yaproperties = find_peaks(-angmom[:,1], height=0, distance=30, prominence=500, width=10)
            xzpeaks, xzproperties = find_peaks(angmom[:,0] + angmom[:,2], height=0, distance=30, prominence=500, width=10)
            xzapeaks, xzaproperties = find_peaks(-(angmom[:,0] + angmom[:,2]), height=0, distance=30, prominence=500, width=10)

            self.features['ypeaks'] = (len(ypeaks) + len(yapeaks)) / self.numframes
            self.features['xzpeaks'] = (len(xzpeaks) + len(xzapeaks)) / self.numframes

            #if there are non NaN values for prominence and width, calculate mean and std
            if np.isnan(yproperties['prominences']).all() == False:
                self.features['yprominence'] = np.mean(yproperties['prominences'])
                self.features['yprominencestd'] = np.std(yproperties['prominences'])
            if np.isnan(yproperties['widths']).all() == False:
                self.features['ywidth'] = np.mean(yproperties['widths'])
                self.features['ywidthstd'] = np.std(yproperties['widths'])
            if np.isnan(xzproperties['prominences']).all() == False:
                self.features['xzprominence'] = np.mean(xzproperties['prominences'])
                self.features['xzprominencestd'] = np.std(xzproperties['prominences'])
            if np.isnan(xzproperties['widths']).all() == False:
                self.features['xzwidth'] = np.mean(xzproperties['widths'])
                self.features['xzwidthstd'] = np.std(xzproperties['widths'])
            
            #for the NaN values, set to 0
            if np.isnan(yproperties['prominences']).all() == True:
                self.features['yprominence'] = 0
                self.features['yprominencestd'] = 0
            if np.isnan(yproperties['widths']).all() == True:
                self.features['ywidth'] = 0
                self.features['ywidthstd'] = 0
            if np.isnan(xzproperties['prominences']).all() == True:
                self.features['xzprominence'] = 0
                self.features['xzprominencestd'] = 0
            if np.isnan(xzproperties['widths']).all() == True:
                self.features['xzwidth'] = 0
                self.features['xzwidthstd'] = 0

            #ankle height
            #compute lowest point of ankle
            floor = min(self.pos[Rankle][:,1]) 
            ankleheight = (self.pos[Lankle][:,1] + self.pos[Rankle][:,1] / 2) - floor
            ankleheightm = ankleheight.mean()
            
            self.features['ankleheight'] = ankleheightm
            self.features['ankleheightstd'] = ankleheight.std()
            
            #hand and ankle distance
            Lwrist = self.pos[7]  #left wrist
            Rwrist = self.pos[8]  #right wrist
            handspace = np.abs(Lwrist - Rwrist).mean(axis=1)

            Rankle = self.pos[14]  #right ankle
            Lankle = self.pos[13]  #left ankle
            anklespace = np.abs(Lankle - Rankle).mean(axis=1)

            contraspace = np.abs(Lankle - Rwrist).mean(axis=1) + np.abs(Rankle - Lwrist).mean(axis=1)
            self.features['contraspace'] = contraspace.mean()
            self.features['contraspacstd'] = contraspace.std()

            #rate of change of anklespace and handspace
            anklespacevel = (anklespace[1:] - anklespace[:-1]) / self.dt
            #savgol filter
            anklespacevel = savgol_filter(anklespacevel, window_length=45, polyorder=2, mode='nearest')
            #acceleration
            anklespaceacc = (anklespacevel[1:] - anklespacevel[:-1]) / self.dt
            handspacevel = (handspace[1:] - handspace[:-1]) / self.dt
            handspacevel = savgol_filter(handspacevel, window_length=45, polyorder=2, mode='nearest')
            handspaceacc = (handspacevel[1:] - handspacevel[:-1]) / self.dt
            
            self.features['anklespaceacc'] = anklespaceacc.mean()
            self.features['Handspaceacc'] = handspaceacc.mean()
            self.features['anklespaceaccstd'] = anklespaceacc.std()
            self.features['Handspaceaccstd'] = handspaceacc.std()


    
    def get_expandedness(self, sparse=False):   #expandedness ~ distance of joints from sacrum
        
        self.get_sacrum()
        
        Dsfromsacrum = np.empty_like(self.pos)  #inits for diff in pos, vel, and accel from sacrum
        Vsfromsacrum = np.empty_like(self.velocity)  
        Asfromsacrum = np.empty_like(self.acceleration)  
        Jsfromsacrum = np.empty_like(self.jerk)
                
        for j in range(self.numjoints):                         #DISTANCE of each joint from sacrum                                    
            Dsfromsacrum[j] = np.abs((self.pos[j] - self.sacrum[0]))
        
        expa = Dsfromsacrum.sum(axis=0)                         #sum over joints to get expandedness per frame per dimension
        expa = expa.sum(axis=1)                                 #sum over dimensions to get expandedness over frames
        
        for j in range(self.numjoints):                         #diff in VELOCITY of each joint from sacrum, same as above
            Vsfromsacrum[j] = np.abs((self.velocity[j] - self.sacrum[1]))
        
        expavel = Vsfromsacrum.sum(axis=0)                      
        expavel = expavel.sum(axis=1)                           

        for j in range(self.numjoints):                         #diff in ACCELERATION of each joint from sacrum, same as above      
            Asfromsacrum[j] = np.abs((self.acceleration[j] - self.sacrum[2]))

        expaacc = Asfromsacrum.sum(axis=0)                      
        expaacc = expaacc.sum(axis=1)                                 

        for j in range(self.numjoints):                         #diff in JERK of each joint from sacrum, same as above      
            Jsfromsacrum[j] = np.abs((self.jerk[j] - self.sacrum[3]))   

        expajer = Jsfromsacrum.sum(axis=0)                      
        expajer = expajer.sum(axis=1)

        self.features['Expandedness'] = expa.sum()/self.numframes
        self.features['Expandedness_std'] = expa.std()

        if sparse==False:
            self.features['Expandedness_range'] = expa.max() - expa.min()
            self.features['Expandednessvel'] = expavel.sum()/self.numframes
            self.features['Expandednessvel_range'] = expavel.max() - expavel.min()
            self.features['Expandednessacc'] = expaacc.sum()/self.numframes
            self.features['Expandednessacc_range'] = expaacc.max() - expaacc.min()
            self.features['Expandednessjer'] = expajer.sum()/self.numframes
            self.features['Expandednessjer_range'] = expajer.max() - expajer.min()

        
              
    def get_asymmetries(self, Ridxs=[4,6,8,10,12,14], Lidxs=[3,5,7,9,11,13], 
        Inidxs=[3, 4, 9, 10], Outidxs=[7, 8, 13, 14],                          #asymmetries ~ difference in joint positions
        Topidxs=[3, 4, 5, 6, 7, 8], Botidxs=[9, 10, 11, 12, 13, 14],
        sparse=False):           #default values for aist++ dataset
                        
                                                                                             
        Rvel, Lvel, Racc, Lacc, Rjer, Ljer = [np.zeros(self.numframes) for i in range(6)]
        
        for j in Ridxs:
            Rvel += np.sum(self.velocity[j], axis=1)            #sum joints over dimensions on each side
            Racc += np.sum(self.acceleration[j], axis=1)
            Rjer += np.sum(self.jerk[j], axis=1)
            
        for j in Lidxs:
            Lvel += np.sum(self.velocity[j], axis=1)
            Lacc += np.sum(self.acceleration[j], axis=1)
            Ljer += np.sum(self.jerk[j], axis=1)

        velratioRL = Rvel / Lvel
        accelratioRL = Racc / Lacc
        jerkratioRL = Rjer / Ljer
              
        if sparse==False:
            self.features['Asym_RL_acc'] = np.sum(accelratioRL)
            self.features['Asym_RL_jer'] = np.sum(jerkratioRL) 
            self.features['Asym_RL_vel'] = np.sum(velratioRL)
        
        velratiomoments = np.split(velratioRL, self.moments)             #split each div by moment = 15frames = 1/4sec
        accelratiomoments = np.split(accelratioRL, self.moments)
        jerkratiomoments = np.split(jerkratioRL, self.moments)
        
        velmeans = [[] for i in range(len(self.moments))]
        accelmeans = [[] for i in range(len(self.moments))]
        jerkmeans = [[] for i in range(len(self.moments))]
        
        for m in range(len(self.moments)):                              #take mean of each moment
            velmeans[m] = np.mean(velratiomoments[m])
            accelmeans[m] = np.mean(accelratiomoments[m])
            jerkmeans[m] = np.mean(jerkratiomoments[m])
            
        if sparse==False:
            self.features['Asym_RL_acc_std'] = np.std(accelmeans)
            self.features['Asym_RL_jer_std'] = np.std(jerkmeans)
            self.features['Asym_RL_vel_std'] = np.std(velmeans)
        
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

        velratioIO = Invel / Outvel
        accelratioIO = Inacc / Outacc
        jerkratioIO = Injer / Outjer
           
        self.features['Asym_IO_acc'] = np.sum(accelratioIO) 
        self.features['Asym_IO_acc_std'] = np.std(accelmeans)

        if sparse==False:
            self.features['Asym_IO_jer'] = np.sum(jerkratioIO) 
            self.features['Asym_IO_vel'] = np.sum(velratioIO)
        
        velratiomoments = np.split(velratioIO, self.moments)             
        accelratiomoments = np.split(accelratioIO, self.moments)
        jerkratiomoments = np.split(jerkratioIO, self.moments)
        
        velmeans = [[] for i in range(len(self.moments))]
        accelmeans = [[] for i in range(len(self.moments))]
        jerkmeans = [[] for i in range(len(self.moments))]
        
        for m in range(len(self.moments)):                              
            velmeans[m] = np.mean(velratiomoments[m])
            accelmeans[m] = np.mean(accelratiomoments[m])
            jerkmeans[m] = np.mean(jerkratiomoments[m])
        

        if sparse==False:
            self.features['Asym_IO_jer_std'] = np.std(jerkmeans)
            self.features['Asym_IO_vel_std'] = np.std(velmeans)
        
        #repeat above for top vs bottom asymmetry
        
        Topvel, Botvel, Topacc, Botacc, Topjer, Botjer = [np.zeros(self.numframes) for i in range(6)]
        
        for j in Topidxs:
            Topvel += np.sum(self.velocity[j], axis=1)            
            Topacc += np.sum(self.acceleration[j], axis=1)
            Topjer += np.sum(self.jerk[j], axis=1)


        for j in Botidxs:
            Botvel += np.sum(self.velocity[j], axis=1)
            Botacc += np.sum(self.acceleration[j], axis=1)
            Botjer += np.sum(self.jerk[j], axis=1)

        velratioTB = Topvel / Botvel
        accelratioTB = Topacc / Botacc
        jerkratioTB = Topjer / Botjer

        if sparse==False:
            self.features['Asym_TB_acc'] = np.sum(accelratioTB)
            self.features['Asym_TB_jer'] = np.sum(jerkratioTB) 
            self.features['Asym_TB_vel'] = np.sum(velratioTB)
        
        velratiomoments = np.split(velratioTB, self.moments)             
        accelratiomoments = np.split(accelratioTB, self.moments)
        jerkratiomoments = np.split(jerkratioTB, self.moments)
        
        velmeans = [[] for i in range(len(self.moments))]
        accelmeans = [[] for i in range(len(self.moments))]
        jerkmeans = [[] for i in range(len(self.moments))]
        
        for m in range(len(self.moments)):                              
            velmeans[m] = np.mean(velratiomoments[m])
            accelmeans[m] = np.mean(accelratiomoments[m])
            jerkmeans[m] = np.mean(jerkratiomoments[m])

        if sparse==False:
            self.features['Asym_TB_acc_std'] = np.std(accelmeans)
            self.features['Asym_TB_jer_std'] = np.std(jerkmeans) 
            self.features['Asym_TB_vel_std'] = np.std(velmeans)

    def get_joint_corr(self, jointaccel1, jointaccel2, label, prominence=.0001, distance=30, sparse=False):
        
        #will look at vertical and horizontal dimensions separately
        #get correlation of acceleration between two joints in a given dimension

        moves1 = [jointaccel1.T[i] for i in range(3)] #list of acceleration data for each joint in each dimension
        moves2 = [jointaccel2.T[i] for i in range(3)]
        
        corry = (np.correlate(moves1[1], moves1[1], mode='full') + np.correlate(moves2[1], moves2[1], mode='full')) / 2
        corry = corry[corry.size//2:] #take only positive lags
        corry = corry / corry[0] #normalize
        corrxz = (np.correlate(moves1[0], moves1[0], mode='full') + np.correlate(moves2[0], moves2[0], mode='full') +
                np.correlate(moves1[2], moves1[2], mode='full') + np.correlate(moves2[2], moves2[2], mode='full')) / 2
        corrxz = corrxz[corrxz.size//2:] #take only positive lags
        corrxz = corrxz / corrxz[0] #normalize

        peaksy, propertiesy = find_peaks(corry, prominence=prominence, distance=distance, height=0)
        lastpeaky = peaksy[-1]/len(corry) #get last peak's time lag
        onehity = propertiesy['peak_heights'][0]
        promstdy = np.std(propertiesy['prominences']) #get standard deviation of prominences
        peak1y = peaksy[np.argsort(propertiesy['peak_heights'])[-1]] / len(corry) #get top peak's time lag
        peak2y = peaksy[np.argsort(propertiesy['peak_heights'])[-2]] / len(corry) #get 2nd peak's time lag
        prom1y = propertiesy['prominences'][np.argsort(propertiesy['peak_heights'])[-1]] #get top peak's prominence
        prom2y = propertiesy['prominences'][np.argsort(propertiesy['peak_heights'])[-2]] #get 2nd peak's prominence
        
        peaksxz, propertiesxz = find_peaks(corrxz, prominence=prominence, distance=distance, height=0)
        lastpeakxz = peaksxz[-1]/len(corrxz) #get last peak's time lag
        onehitxz = propertiesxz['peak_heights'][0]
        promstdxz = np.std(propertiesxz['prominences'])
        peak1xz = peaksxz[np.argsort(propertiesxz['peak_heights'])[-1]] / len(corrxz) #get top peak's time lag
        peak2xz = peaksxz[np.argsort(propertiesxz['peak_heights'])[-2]] / len(corrxz) #get 2nd peak's time lag
        prom1xz = propertiesxz['prominences'][np.argsort(propertiesxz['peak_heights'])[-1]] #get top peak's prominence
        prom2xz = propertiesxz['prominences'][np.argsort(propertiesxz['peak_heights'])[-2]] #get 2nd peak's prominence

        
        
        self.features['lastpeak_y{}'.format(label)] = lastpeaky
        self.features['prominence1_y{}'.format(label)] = prom1y
        
        if sparse==False:
            try:
                peak3y = peaksy[np.argsort(propertiesy['peak_heights'])[-3]] / len(corry) #get 3rd peak's time lag
            except IndexError:
                peak3y = 0
            try:
                prom3y = propertiesy['prominences'][np.argsort(propertiesy['peak_heights'])[-3]] #get 3rd peak's prominence
            except IndexError:
                prom3y = 0
            try:
                peak3xz = peaksxz[np.argsort(propertiesxz['peak_heights'])[-3]] / len(corrxz) #get 3rd peak's time lag
            except IndexError:
                peak3xz = 0
            try:
                prom3xz = propertiesxz['prominences'][np.argsort(propertiesxz['peak_heights'])[-3]] #get 3rd peak's prominence
            except IndexError:
                prom3xz = 0

            self.features['peak1_y{}'.format(label)] = peak1y
            self.features['lastpeak_xz{}'.format(label)] = lastpeakxz
            self.features['onehit_y{}'.format(label)] = onehity
            self.features['onehit_xz{}'.format(label)] = onehitxz
            self.features['promstd_xz{}'.format(label)] = promstdxz
            self.features['promstd_y{}'.format(label)] = promstdy
            self.features['peak2_y{}'.format(label)] = peak2y
            self.features['prominence2_y{}'.format(label)] = prom2y
            self.features['peak3_y{}'.format(label)] = peak3y
            self.features['prominence3_y{}'.format(label)] = prom3y
            self.features['peak1_xz{}'.format(label)] = peak1xz
            self.features['prominence1_xz{}'.format(label)] = prom1xz
            self.features['peak2_xz{}'.format(label)] = peak2xz
            self.features['prominence2_xz{}'.format(label)] = prom2xz
            self.features['peak3_xz{}'.format(label)] = peak3xz
            self.features['prominence3_xz{}'.format(label)] = prom3xz
            
        #this is really for 2d data, below
        # if sparse==True:
        #     corrx = (np.correlate(moves1[0], moves1[0], mode='full') + np.correlate(moves2[0], moves2[0], mode='full')) / 2
        #     corrx = corrx[corrx.size//2:] #take only positive lags
        #     corrx = corrx / corrx[0] #normalize

        #     peaksx, propertiesx = find_peaks(corrx, prominence=prominence, distance=distance, height=0)
        #     lastpeakx = peaksx[-1]
        #     onehitx = propertiesx['peak_heights'][0]
        #     deviatex = np.std(corrx[:lastpeakx]) / lastpeakx
        #     peak1x = peaksx[np.argsort(propertiesx['peak_heights'])[-1]] / len(x) #get top peak's time lag
        #     prom1x = propertiesx['prominences'][np.argsort(propertiesx['peak_heights'])[-1]] #get top peak's prominence

        #     self.features['lastpeak_x{}'.format(label)] = lastpeakx
        #     self.features['onehit_x{}'.format(label)] = onehitx
        #     self.features['deviate_x{}'.format(label)] = deviatex
        #     self.features['peak1_x{}'.format(label)] = peak1x
        #     self.features['prominence1_x{}'.format(label)] = prom1x

            

    def get_joint_corr_features(self, sparse):

        #setting joint accelerations to variables for get_joint_corr
        self.get_sacrum()
        nose = self.acceleration[0]
        Lshoulder = self.acceleration[3]
        Rshoulder = self.acceleration[4]
        Lelbow = self.acceleration[5]
        Relbow = self.acceleration[6]
        Lwrist = self.acceleration[7]
        Rwrist = self.acceleration[8]
        Lhip = self.acceleration[9]
        Rhip = self.acceleration[10]
        Lknee = self.acceleration[11]
        Rknee = self.acceleration[12]
        Lankle = self.acceleration[13]
        Rankle = self.acceleration[14]
        sacrum = self.sacrum[2]
        
        joints = [nose, Lshoulder, Rshoulder, Lelbow, Relbow, Lwrist, Rwrist, Lhip, Rhip, Lknee, Rknee, Lankle, Rankle, sacrum]

        
        if sparse==True:
            self.get_joint_corr(Rwrist, Lwrist, 'wrists')
            self.get_joint_corr(Rshoulder, Lknee, 'contralatRsLk')
            self.get_joint_corr(Rankle, Lankle, 'ankles')

        
        if sparse==False:
            self.get_joint_corr(nose, nose, 'nose')
            self.get_joint_corr(Rwrist, Rwrist, 'Rwrist')
            self.get_joint_corr(Lankle, Lankle, 'Lankle')
            self.get_joint_corr(Rankle, Rankle, 'Rankle')
            self.get_joint_corr(Rankle, Lankle, 'ankles')
            self.get_joint_corr(sacrum, sacrum, 'sacrum')
            self.get_joint_corr(Rshoulder, Lknee, 'contralatRsLk')
            self.get_joint_corr(Lhip, Rwrist, 'contralatLhRw')
            self.get_joint_corr(Relbow, Lelbow, 'elbows')
            self.get_joint_corr(Lwrist, Lwrist, 'Lwrist')
            self.get_joint_corr(Rknee, Lknee, 'knees')
            self.get_joint_corr(Rwrist, Rankle, 'RwristRankle')
            self.get_joint_corr(Lwrist, Lankle, 'LwristLankle')
            self.get_joint_corr(Rwrist, Lwrist, 'wrists')


                 
            

    def get_features(self, sparse=False):
        self.get_movedata()
        self.features['id'] = self.id
        self.features['Genre'] = self.genre
        self.get_mofeatures(sparse=sparse)
        self.get_expandedness(sparse=sparse)
        self.get_asymmetries(sparse=sparse)
        self.get_joint_corr_features(sparse=sparse)

     
        
        

        
