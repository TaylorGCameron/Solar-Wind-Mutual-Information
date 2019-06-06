# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:07:35 2019

@author: Taylor
"""
import mutual_information.data as data
import os

#process the SuperMAG text files
data.process_SuperMAG()

#Pull the ACE data from CDAWeb
data.pull_data()

#create index lists for quick data averaging
data.calc_time_indices()

#Average ACE and SuperMAG
data.average_ACE()
data.average_SuperMAG()

#calculate MVAB0_normals
data.calc_MVAB0_normals()

#at this point it should be okay to delete the ACE and SuperMAG data if space is an issue.

d = data.load_data()