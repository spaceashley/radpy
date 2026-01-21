import os

raddir = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(raddir, 'data')
# All Least squares method
hapath = os.path.join(datadir, 'ldcHA.csv')
hppath = os.path.join(datadir, 'ldcHP.csv')
japath = os.path.join(datadir, 'ldcJA.csv')
jppath = os.path.join(datadir, 'ldcJP.csv')
kapath = os.path.join(datadir, 'ldcKA.csv')
kppath = os.path.join(datadir, 'ldcKP.csv')
rapath = os.path.join(datadir, 'ldcRA.csv')
rppath = os.path.join(datadir, 'ldcRP.csv')

# All flux conservation method
hafpath = os.path.join(datadir, 'ldcHAF.csv')
hpfpath = os.path.join(datadir, 'ldcHPF.csv')
jafpath = os.path.join(datadir, 'ldcJAF.csv')
jpfpath = os.path.join(datadir, 'ldcJPF.csv')
kafpath = os.path.join(datadir, 'ldcKAF.csv')
kpfpath = os.path.join(datadir, 'ldcKPF.csv')
rafpath = os.path.join(datadir, 'ldcRAF.csv')
rpfpath = os.path.join(datadir, 'ldcRPF.csv')


classicpath = os.path.join(datadir, 'ClassicData.csv')
pavopath = os.path.join(datadir, 'PAVOdata.csv')
vegapath = os.path.join(datadir, 'Vegadata.csv')

svopath = os.path.join(datadir, 'svo_filter_info.csv')
