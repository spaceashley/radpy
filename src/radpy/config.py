import os

raddir = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(raddir, 'data')

ldckpath = os.path.join(datadir, 'ldc_k_table.csv')
ldcipath = os.path.join(datadir, 'ldc_I_table.csv')
ldcrpath = os.path.join(datadir, 'ldctable.csv')

classicpath = os.path.join(datadir, 'ClassicData.csv')
pavopath = os.path.join(datadir, 'PAVOdata.csv')
vegapath = os.path.join(datadir, 'Vegadata.csv')

if __name__ == '__main__':
    import pandas as pd
    print(os.path.isfile(ldcrpath))
    ldc_dataR = pd.read_csv(ldcrpath)
    tempsR = ldc_dataR['Teff'].tolist()