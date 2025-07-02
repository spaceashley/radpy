import os

raddir = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(raddir, 'data')

hapath = os.path.join(datadir, 'ldcHA.csv')
hppath = os.path.join(datadir, 'ldcHP.csv')
japath = os.path.join(datadir, 'ldcJA.csv')
jppath = os.path.join(datadir, 'ldcJP.csv')
kapath = os.path.join(datadir, 'ldcKA.csv')
kppath = os.path.join(datadir, 'ldcKP.csv')
rapath = os.path.join(datadir, 'ldcRA.csv')
rppath = os.path.join(datadir, 'ldcRP.csv')

classicpath = os.path.join(datadir, 'ClassicData.csv')
pavopath = os.path.join(datadir, 'PAVOdata.csv')
vegapath = os.path.join(datadir, 'Vegadata.csv')

if __name__ == '__main__':
    import pandas as pd
    print(os.path.isfile(ldcrpath))
    ldc_dataR = pd.read_csv(ldcrpath)
    tempsR = ldc_dataR['Teff'].tolist()