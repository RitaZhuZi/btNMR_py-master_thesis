#!/usr/bin/env python3

import csv
import random
import numpy as np
import pandas as pd
import tensorflow as tf

evolution_time = [0.02, 0.07] # evolution time for each spin echo spectra. unit is (ms). 
input_field = 80 # LF initial field strength
target_field = 400 # HF target field strength

'''
Small molecule spectra parameters definition:

ppm_list = [] # spins of the same species should be input separately e.g. [6.544,6.544,76.544]
j_list = [[]] # 2D matrix with diagonal as 0. e.g. [[0    ,3.481,2.080],[0    ,0    ,0.953],[0    ,0    ,0    ],]
spin_num = 0 # Number of spins in total. Should be the same as ppm_list length
'''

'''
Peptide amide proton spectra parameters definition:

resi_num = 0 # number of residues presenting in the spectrum
residue_lib_file = './residue_lib.csv' # pathway of storing BMRB residual chemical shift library
residue_lib = pd.read_csv(residue_lib_file)
hn = residue_lib[residue_lib['atom_id'] == 'H']
'''

'''
random training spectra parameters definition:

max_spin_num = 0 # maximum of number of spins presenting in the spectra
j_num = 0 # maximum of coupling for all spins
'''

class Generator():
    def __init__(self, spin_num, noise_mean=0, noise_stdv=0.05, 
    ppm=[-10, 10],
    r2=[1, 3],
    j=[2, 10],
    sw=24,
    phase=[-10, 10],
    spec_length=[1024, 6000],
    input_field_strength=input_field,
    target_field_strength=target_field):
        self.spin_num = spin_num 
        self.noise_mean = noise_mean
        self.noise_stdv = noise_stdv
        self.ppm_range = ppm
        self.r2_range = r2
        self.j_range = j
        self.sw = sw
        self.phase_range = phase
        self.spec_length = spec_length
        self.input_field_strength = input_field_strength
        self.target_field_strength = target_field_strength

    def small_molecule_parameters(self):
        ns = random.randint(self.spec_length[0], self.spec_length[1]+1)
        if ns > max_ns:
            ns = max_ns
        self.ns = ns
        self.input_interval = 1 / (self.input_field_strength * self.sw)
        self.target_interval = 1 / (self.target_field_strength * self.sw)
        self.input_time = tf.reshape(tf.range(0, self.ns, dtype=tf.float32) * self.input_interval, (1,-1))   #shape=(1,1024)
        self.target_time = tf.reshape(tf.range(0, self.ns, dtype=tf.float32) * self.target_interval, (1,-1))   #shape=(1,1024)
        self.ppm_list = tf.reshape(tf.constant(ppm_list), (-1,1))   #frequencies for each spins
        self.r2_list = tf.math.abs(tf.random.uniform(shape=(self.spin_num,1), minval=self.r2_range[0], maxval=self.r2_range[1], dtype=tf.float32))   #Random r2 for each spin
        self.phase_list = tf.math.abs(tf.random.normal(shape=(self.spin_num,1), mean=(self.phase_range[0] + self.phase_range[1]) / 2, stddev=self.phase_range[1] / 4, dtype=tf.float32))   #Random phase for each spin
        j = tf.constant(j_list)
        self.j_list = j + tf.transpose(j)
        self.phase_list = tf.cast(self.phase_list, dtype=tf.complex64)
        self.j_list = tf.expand_dims(self.j_list, axis=-1) #shape=(spin,spin,1)

    def amide_proton_parameters(self):
        ns = random.randint(self.spec_length[0], self.spec_length[1]+1)
        if ns > max_ns:
            ns = max_ns
        self.ns = ns
        self.input_interval = 1 / (self.input_field_strength * self.sw)
        self.target_interval = 1 / (self.target_field_strength * self.sw)
        self.input_time = tf.reshape(tf.range(0, self.ns, dtype=tf.float32) * self.input_interval, (1,-1))   #shape=(1,1024)
        self.target_time = tf.reshape(tf.range(0, self.ns, dtype=tf.float32) * self.target_interval, (1,-1))   #shape=(1,1024)
        self.random_resi = tf.random.uniform(shape=(self.resi_num,), minval=0, maxval=20, dtype=tf.int32)
        residue_list = hn.iloc[self.random_resi]
        print(residue_list['comp_id'])
        self.cs_list = tf.constant([np.random.normal(size=(1,), loc=residue_list['avg'].iloc[i], scale=residue_list['std'].iloc[i]) for i in range(self.resi_num)], shape=(self.resi_num,1), dtype=tf.float32)
        print(self.cs_list)
        mask = tf.constant(residue_list['comp_id'] == 'GLY', shape=(self.resi_num,1), dtype=tf.float32)
        self.j_list = tf.constant(np.random.normal(size=(self.resi_num, 1), loc=7.0, scale=1.0), dtype=tf.float32)
        self.j_list = tf.concat((self.j_list,self.j_list*mask), axis=1)
        self.j_list = tf.expand_dims(self.j_list, axis=-1) #shape=(spin,spin,1)
        self.r2_list = tf.math.abs(tf.random.uniform(shape=(self.resi_num,1), minval=self.r2_range[0], maxval=self.r2_range[1], dtype=tf.float32))   #Random r2 for each spin

    def random_training_parameters(self):
        ns = random.randint(self.spec_length[0], self.spec_length[1]+1)
        if ns > max_ns:
            ns = max_ns
        self.ns = ns
        self.input_interval = 1 / (self.input_field_strength * self.sw)
        self.target_interval = 1 / (self.target_field_strength * self.sw)
        self.input_time = tf.reshape(tf.range(0, self.ns, dtype=tf.float32) * self.input_interval, (1,-1))   #shape=(1,1024)
        self.target_time = tf.reshape(tf.range(0, self.ns, dtype=tf.float32) * self.target_interval, (1,-1))   #shape=(1,1024)
        self.ppm_list = tf.math.abs(tf.random.uniform(shape=(self.spin_num,1), minval=self.ppm_range[0], maxval=self.ppm_range[1], dtype=tf.float32))   #Randome frequencies for each spins
        self.r2_list = tf.math.abs(tf.random.uniform(shape=(self.spin_num,1), minval=self.r2_range[0], maxval=self.r2_range[1], dtype=tf.float32))   #Random r2 for each spin
        j = tf.random.uniform(shape=(self.spin_num,self.spin_num), minval=self.j_range[0], maxval=self.j_range[1], dtype=tf.float32)   #Random Js regarding each two spins
        j_mask = tf.random.uniform((self.spin_num,self.spin_num), maxval=2, minval=0, dtype=tf.int32)
        diagonal = tf.linalg.diag_part(j_mask)
        diagonal = tf.linalg.diag(diagonal)
        j_mask = j_mask - diagonal
        j_mask = tf.where(tf.less(tf.cumsum(j_mask, axis=0), j_num + 1),
            j_mask,
            tf.zeros_like(j_mask))
        j_mask = tf.where(tf.less(tf.cumsum(j_mask, axis=1), j_num + 1),
            j_mask,
            tf.zeros_like(j_mask))
        j = j * tf.cast(j_mask, dtype=tf.float32)
        j = tf.linalg.band_part(j, 0, -1)
        self.j_list = j + tf.transpose(j)
        self.j_list = tf.expand_dims(self.j_list, axis=-1) #shape=(spin,spin,1)


    def get__fid(self):
        fid = tf.complex(tf.random.uniform(shape=(self.spin_num, self.ns)),0.)*tf.complex(0.,0.)
        freq_time = tf.cast(self.ppm_list * self.input_field_strength * self.input_time, dtype=tf.complex64)
        r2_time = tf.cast(-self.r2_list * self.input_time, dtype=tf.complex64)
        phase = tf.math.exp(1j * self.phase_list * pi / 180)   #Phase
        time_coupling = tf.expand_dims(self.input_time, axis=0) #shape=(1,1,1024)
        coupling = tf.cast(tf.math.reduce_prod(tf.math.cos(pi * time_coupling * self.j_list), axis=1), dtype=tf.complex64)
        fid = (tf.math.exp(1j * 2 * pi * freq_time) 
            * tf.math.exp(r2_time)
            * phase
            * coupling)
        fid = tf.reshape(tf.math.reduce_sum(fid, axis=0), (1,-1))
        noise = tf.random.normal(shape=(self.ns,), mean=self.noise_mean, stddev=self.noise_stdv, dtype=tf.float32)   #Random noise, normal distribution
        window_func = 1.0 * tf.math.exp(-(self.input_time - 0.0) ** 2 / (2 * (0.5 * self.input_time[0][-1]) ** 2))   #Apodisation
        fid = (fid + tf.cast(noise, dtype=tf.complex64)) * tf.cast(window_func, dtype=tf.complex64)
        zf_fid = tf.pad(fid,[[0,0],[0,max_ns-self.ns]])
        zf_fid = tf.reshape(tf.stack([tf.math.real(zf_fid), tf.math.imag(zf_fid)], axis=-1), (2*max_ns,1))
        zf_fid = zf_fid / tf.math.reduce_max(tf.math.abs(zf_fid))
        return zf_fid


    def get_target_fid(self):
        fid = tf.complex(tf.random.uniform(shape=(self.spin_num, self.ns)),0.)*tf.complex(0.,0.)
        freq_time = tf.cast(self.ppm_list * self.target_field_strength * self.target_time, dtype=tf.complex64)
        r2_time = tf.cast(-self.r2_list * self.target_time, dtype=tf.complex64)
        time_coupling = tf.expand_dims(self.target_time, axis=0)
        coupling = tf.cast(tf.math.reduce_prod(tf.math.cos(pi * time_coupling * self.j_list), axis=1), dtype=tf.complex64)
        fid = (tf.math.exp(1j * 2 * pi * freq_time)
            * tf.math.exp(r2_time)
            * coupling)
        fid = tf.reshape(tf.math.reduce_sum(fid, axis=0), (1,-1))
        window_func = 1.0 * tf.math.exp(-(self.target_time - 0.0) ** 2 / (2 * (0.5 * self.target_time[0][-1]) ** 2))   #Apodisation
        fid = fid * tf.cast(window_func, dtype=tf.complex64)
        zf_fid = tf.pad(fid,[[0,0],[0,max_ns-self.ns]])
        zf_fid = tf.reshape(tf.stack([tf.math.real(zf_fid), tf.math.imag(zf_fid)], axis=-1), (2*max_ns,1))
        zf_fid = zf_fid / tf.math.reduce_max(tf.math.abs(zf_fid))
        uncertainty = tf.zeros_like(zf_fid, dtype=tf.float32)
        zf_fid = tf.concat([zf_fid, uncertainty], axis=-1)
        return zf_fid


    def get_spin_echo(self, evolution_time):
        fid = tf.complex(tf.random.uniform(shape=(self.spin_num, self.ns)),0.)*tf.complex(0.,0.)
        freq_time = tf.cast(self.ppm_list * self.input_field_strength * self.input_time, dtype=tf.complex64)
        r2_time = tf.cast(-self.r2_list * self.input_time, dtype=tf.complex64)
        phase = tf.math.exp(1j * self.phase_list * pi / 180)   #Phase
        time_coupling = tf.expand_dims(self.input_time + evolution_time, axis=0) #shape=(1,1,1024)
        coupling = tf.cast(tf.math.reduce_prod(tf.math.cos(pi * time_coupling * self.j_list), axis=1), dtype=tf.complex64)
        fid = (tf.math.exp(1j * 2 * pi * freq_time) 
            * tf.math.exp(r2_time)
            * phase
            * coupling)
        fid = tf.reshape(tf.math.reduce_sum(fid, axis=0), (1,-1))
        noise = tf.random.normal(shape=(self.ns,), mean=self.noise_mean, stddev=self.noise_stdv, dtype=tf.float32)   #Random noise, normal distribution
        window_func = 1.0 * tf.math.exp(-(self.input_time - 0.0) ** 2 / (2 * (0.5 * self.input_time[0][-1]) ** 2))   #Apodisation
        fid = (fid + tf.cast(noise, dtype=tf.complex64)) * tf.cast(window_func, dtype=tf.complex64)
        zf_fid = tf.pad(fid,[[0,0],[0,max_ns-self.ns]])
        zf_fid = tf.reshape(tf.stack([tf.math.real(zf_fid), tf.math.imag(zf_fid)], axis=-1), (2*max_ns,1))
        zf_fid = zf_fid / tf.math.reduce_max(tf.math.abs(zf_fid))
        return zf_fid



