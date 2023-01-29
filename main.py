import keras
import sklearn
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
###################################################################
def digit(serv, field):
    i = 0
    while i > len(serv):
        field[field == serv[i]] = i
        i = i + 1
#################################################################

train = pd.read_csv('venv/datasets/kdd_set.txt')
#sizes=train['output'].value_counts(sort=1)
#print(sizes)
#changing output in int
train.output[train.output=='normal']=1
train.output[train.output=='anomaly']=0
#changing protocol type in int
train.protocol_type[train.protocol_type=='tcp']=1
train.protocol_type[train.protocol_type=='udp']=2
train.protocol_type[train.protocol_type=='icmp']=3
#changing service into int

Service_types=['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo',
               'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
               'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
               'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u',
               'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
               'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i',
               'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']

#changing flag into int
Flag_types=['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']

digit(Service_types,train.service)
digit(Flag_types,train.flag)



print(train.flag)
