#Example basic usage catalog class, __version__=0.2
#last updated: 2020-12-05 

from catalog import catalog
import matplotlib.pyplot as plt

#/home/silvia/Dropbox/Documenti/work/gammarays/1pPDF/Bulge/
mycatalog=catalog('gll_psc_v24.fit', 1)
mycatalog()

#Print vector with list of features
print('List of features names', mycatalog.features_names)

#Select one feature array
listfluxes=mycatalog.feature('Flux1000')
print(listfluxes)

#Stats info for a feature
mycatalog.feat_stats('Flux1000')

#Returns specific info for a feature
print(mycatalog.feat_info('GLAT'))

#Basic 1D histogram plot and save (not enough elastic to change in scale, inputs.. maybe remove)
mycatalog.feat_hist('GLAT', None, '1D')

#Basic 2D histogram plot and save
mycatalog.feat_hist('GLAT', 'GLON', '2D')


#Feature selection according to source class, and plot
psrglat= mycatalog.sort_feat('GLAT',mycatalog.class_psr)
agnglat= mycatalog.sort_feat('GLAT',mycatalog.class_agn)
unkglat= mycatalog.sort_feat('GLAT',mycatalog.class_unk)

plt.hist(agnglat, bins='auto', label='AGN')
plt.hist(unkglat, bins='auto', label='UNK')
plt.hist(psrglat, bins='auto', label='PSR')
plt.xlabel('GLAT')
plt.ylabel('# of sources')
plt.legend()
plt.savefig('myhist.png')
