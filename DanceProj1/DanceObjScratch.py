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

        #calculate footdistance
        Lankle = self.pos[13]  #left ankle
        Rankle = self.pos[14]  #right ankle
        footspace = np.abs(Lankle - Rankle).mean(axis=1)
        Lanklex = Lankle[:,0]
        Ranklex = Rankle[:,0]
        Lankley = Lankle[:,1]
        Rankley = Rankle[:,1]
        Lanklez = Lankle[:,2]
        Ranklez = Rankle[:,2]
        footspacex = np.abs(Lanklex - Ranklex)
        footspacey = np.abs(Lankley - Rankley)
        footspacez = np.abs(Lanklez - Ranklez)

        
        #
        #fix footspace?
        self.features['FootspaceX'] = footspacex.mean()
        


        if sparse==False:
            self.features['Expandednessvel_range'] = expavel.max() - expavel.min()
            self.features['Expandednessvel'] = expavel.sum()/self.numframes
            self.features['Expandednessacc'] = expaacc.sum()/self.numframes
            self.features['Expandednessjer'] = expajer.sum()/self.numframes
            self.features['Expandednessacc_range'] = expaacc.max() - expaacc.min()
            self.features['Expandednessjer'] = expajer.sum()/self.numframes
            self.features['Expandednessjer_range'] = expajer.max() - expajer.min()
    
            self.features['Footspace_range'] = footspace.max() - footspace.min()
            self.features['FootspaceY'] = footspacey.mean()
            self.features['FootspaceZ'] = footspacez.mean()

        
              
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
              
        self.features['Asym_RL_acc'] = np.sum(accelratioRL) 

        if sparse==False:
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
            self.features['Asym_IO_acc_std'] = np.std(accelmeans)
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
              
        self.features['Asym_TB_acc'] = np.sum(accelratioTB) 

        if sparse==False:
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

    def get_joint_corr(self, jointaccel1, jointaccel2, label, dim, prominence=.0001, distance=30, sparse=False):
        
        #change to autocorrelations, summed and split in two


        #get correlation of acceleration between two joints in a given dimension
        move1 = jointaccel1.T[dim]
        move2 = jointaccel2.T[dim]
        
        x = (np.correlate(move1, move1, mode='full') + np.correlate(move2, move2, mode='full')) / 2
        x = x[x.size//2:] #take only positive lags
        x = x / x[0] #normalize

        peaks, properties = find_peaks(x, prominence=prominence, distance=distance, height=0)
        lastpeak = peaks[-1]
        onehit = properties['peak_heights'][0]
        deviate = np.std(x[:lastpeak]) / lastpeak
        peak1 = peaks[np.argsort(properties['peak_heights'])[-1]] / len(x) #get top peak's time lag
        peak2 = peaks[np.argsort(properties['peak_heights'])[-2]] / len(x) #get 2nd peak's time lag
        prom1 = properties['prominences'][np.argsort(properties['peak_heights'])[-1]] #get top peak's prominence
        prom2 = properties['prominences'][np.argsort(properties['peak_heights'])[-2]] #get 2nd peak's prominence
        dimlabel = ['x', 'y', 'z'][dim]

        self.features['corr_prominence1_{}_{}'.format(label, dimlabel)] = prom1
        self.features['corr_peak1_{}_{}'.format(label, dimlabel)] = peak1
        self.features['corr_onehit_{}_{}'.format(label, dimlabel)] = onehit
        self.features['corr_lastpeak_{}_{}'.format(label, dimlabel)] = lastpeak
       

        if sparse==False:
            try:
                peak3 = peaks[np.argsort(properties['peak_heights'])[-3]] / len(x) #get 3rd peak's time lag
            except IndexError:
                peak3 = 0
            try:
                prom3 = properties['prominences'][np.argsort(properties['peak_heights'])[-3]] #get 3rd peak's prominence
            except IndexError:
                prom3 = 0

            self.features['corr_peak2_{}_{}'.format(label, dimlabel)] = peak2
            self.features['corr_peak3_{}_{}'.format(label, dimlabel)] = peak3
            self.features['corr_prominence2_{}_{}'.format(label, dimlabel)] = prom2
            self.features['corr_prominence3_{}_{}'.format(label, dimlabel)] = prom3
            self.features['corr_deviate_{}_{}'.format(label, dimlabel)] = deviate


            

    def get_joint_corr_features(self, sparse=False):

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

        #get correlation between set in x, y. optional z and more joints
        
        if sparse==True:
            for dim in range(2):
                self.get_joint_corr(nose, nose, 'nose', dim)
                self.get_joint_corr(Rwrist, Lwrist, 'wrists', dim)
                self.get_joint_corr(Rwrist, Lknee, 'RwristLknee', dim)
                self.get_joint_corr(Rshoulder, Lhip, 'RshoLhip', dim)
                self.get_joint_corr(Rankle, Rankle, 'Rankle', dim)

        
        if sparse==False:
            for dim in range(3):
                self.get_joint_corr(nose, nose, 'nose', dim)
                self.get_joint_corr(Rwrist, Rwrist, 'Rwrist', dim)
                self.get_joint_corr(Lwrist, Lwrist, 'Lwrist', dim)
                self.get_joint_corr(Rankle, Rankle, 'Rankle', dim)
                self.get_joint_corr(Lankle, Lankle, 'Lankle', dim)
                self.get_joint_corr(sacrum, sacrum, 'sacrum', dim)
                self.get_joint_corr(Rshoulder, Lhip, 'RshoLhip', dim)
                self.get_joint_corr(Lshoulder, Rhip, 'LshoRhip', dim)
                self.get_joint_corr(Relbow, Relbow, 'Relbow', dim)
                self.get_joint_corr(Lelbow, Lelbow, 'Lelbow', dim)
                self.get_joint_corr(Rknee, Rknee, 'Rknee', dim)
                self.get_joint_corr(Lknee, Lknee, 'Lknee', dim)
                self.get_joint_corr(Rwrist, Rankle, 'RwristRankle', dim)
                self.get_joint_corr(Rshoulder, Lankle, 'RshoLankl', dim)
                #self.get_joint_corr(nose, Lhip, 'noseLhip', dim)
                #self.get_joint_corr(nose, Rankle, 'noseRankle', dim)

                 
            

    def get_features(self, sparse=False):
        self.get_movedata()
        self.features['id'] = self.id
        self.features['Genre'] = self.genre

        self.get_expandedness(sparse=sparse)
        self.get_asymmetries(sparse=sparse)
        self.get_joint_corr_features(sparse=sparse)

     
        
        

        
