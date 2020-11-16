 
"""
Class for reading Fermi-LAT catalog fits files
Version History:
----------------
last updated: 2020-10-24 
usage: see example_catalog.py

updated with MSP-yng classification of pulsars from Fermi-LAT PSR catalog v38
"""

__author__= 'manconi@physik.rwth-aachen.de'
__version__=0.3



from math import *
import matplotlib.pyplot as plt
import numpy as np 
from astropy.io import fits
from astropy.table import Table, Column
import astropy.units as u
from astropy.coordinates import SkyCoord
from timeit import default_timer as timer
import pickle as pkl



class catalog(object):
  
  def __init__(self, filename,  hdn, **kwargs):
    """
      Initialize class instance.
      
      Parameters:
      filename: path to filename of the catalog, expected .fits file type
      hdn: hdn to read in the .fits file 
      
      kwargs:
      
    """
      
    self.filename= filename
    self.hdn= hdn
    
    #Basic catalog features 
    self.Ns=0       #total number of sources
    self.Nfeat=0    #total number of features, 84 expected for 4FGL

    kwargs.setdefault('srclasses',['PSR', 'PWN', 'SNR', 'SPP','GLC', 'SFR','HMB', 'LMB', 'HMB', 'BIN', 'NOV', 'BLL', 'FSRQ', 'RDG', 'AGN','SSRQ', 'CSS', 'BCU', 'NLSY1', 'SEY', 'SBG', 'GAL', 'UNK', 'psr', 'pwn', 'snr', 'spp','glc', 'sfr','hmb', 'lmb', 'hmb', 'bin', 'nov', 'bll', 'fsrq', 'rdg', 'agn', 'ssrq', 'css', 'bcu', 'nlsy1', 'sey', 'sbg', 'gal', 'unk', 'mc',' ', ''])
    srclasses=kwargs['srclasses']    

    class_gal=['PSR', 'PWN', 'SNR', 'SPP','GLC', 'SFR','HMB', 'LMB', 'HMB', 'BIN', 'NOV', 'psr', 'pwn', 'snr', 'spp','glc', 'sfr','hmb', 'lmb', 'hmb', 'bin', 'nov']
    class_extragal=['BLL', 'FSRQ', 'RDG', 'AGN','SSRQ', 'CSS', 'BCU', 'NLSY1', 'SEY', 'SBG', 'GAL','bll', 'fsrq', 'rdg', 'agn','ssrq', 'css', 'bcu', 'nlsy1', 'sey', 'sbg', 'gal', 'mc']

    self.class_unk=['UNK','unk',' ', '']
    self.class_agn=['BLL', 'FSRQ', 'RDG', 'AGN','SSRQ', 'CSS', 'BCU','bll', 'fsrq', 'rdg', 'agn','ssrq', 'css', 'bcu']
    self.class_psr=['PSR', 'psr']



  
  def __call__(self):
    """
     Instance call
     
     Read all the features
     """
     
    #open file, reads data and header, and close file
    hduls= fits.open(self.filename)
    self.head=hduls[self.hdn].header
    self.catdata=hduls[self.hdn].data    
    hduls.close()

    #Basic Processing of data
    self.source_name=self.catdata.field('Source_Name')
    self.Ns=len(self.source_name)
    self.Nfeat=len(self.catdata.dtype.names)
    print('This catalog has '+str(self.Ns)+' sources with '+str(self.Nfeat)+' features')
    
    #Catalog handling with Table, Columns astropy objects
    self.cat_table=Table(self.catdata)
    names = [name for name in self.cat_table.colnames if len(self.cat_table[name].shape) <= 1]
    self.pdTable = self.cat_table[names].to_pandas()
    self.features_names=self.cat_table.colnames
    
    #basic output handling, for reference
    #print(self.cat_table['Flags'].info.unit) 
    #availabe in .info:  dtype, unit, format, description, class, n_bad, lenght
    #print(self.cat_table['Flags'].info('stats'))
    #print(self.cat_table['PL_Index'][3])


    #Reading Fermi-LAT pulsar catalog for MSP-YNG classification
    fn= open("fermi_msp_yng_v38.pkl", "rb") 
    t= pkl.load(fn)
    fn.close()
    msp=t['MSP']
    yng=t['YNG']
    
    #Define a psr_subclass label for each catalog source. 
    #NAP if source is not a PSR, then MSP or YNG according to pulsar catalog
    psr_sc=[]
    cat_msp=[] #for checking with catalog
    cat_yng=[]
    for i in range(self.Ns):
             local_name=str(self.pdTable['ASSOC1'][i])
             if(local_name[4:].strip() in msp): 
                psr_sc.append('MSP')
                cat_msp.append(local_name[4:].strip())
             elif(local_name[4:].strip() in yng): 
                psr_sc.append('YNG')
                cat_yng.append(local_name[4:].strip())
             else: psr_sc.append('NAP')
    self.psr_subclass=psr_sc
    print('Cross-match with Fermi-LAT PSR cat v.38:', psr_sc.count('NAP'), 
          'not a pulsars;', psr_sc.count('YNG'), 'Young PSR, and', psr_sc.count('MSP'), 'MSP')		
   
    #This was to indentify the MSP and YNG in the PSR catalog which do not cross match
    #There is a number (12 yng and 11 msp)
    #for i in range(len(msp)): 
    #  if(msp[i].strip() not in cat_msp): print(msp[i])	
    #for i in range(len(yng)): 
    #  if(yng[i].strip() not in cat_yng): print(yng[i])	


    
  def feature(self, feature):
    return self.cat_table[feature]


  def feat_stats(self, feature):
    #Prints the stats for a given feature in the catalog. Just an interface with astropy utils
    #feature= one of the available features, see self.feature_names
    if(feature in self.features_names): 
           print('Stats Info for', feature)
           return self.cat_table[feature].info('stats')
    else:
           raise ValueError
           print('Selected feature not available, see catalog.features_names output')    
  

  def feat_info(self,feature):
    #Returns the particular info required for the selected feature
    #Available features in catalog.features_names
    if(feature in self.features_names): 
              print('Info for', feature)
              return self.cat_table[feature].info
          
    else:
           raise ValueError
           print('Selected feature not available, see list: catalog.features_names ')    


  def feat_hist(self, feature1,feature2,dimension):
    #Computes and saves the histogram for the asked feature, using auto bins
    #TO DO: Add check for feature type, this only valid for dummy features, not valid for arrays
    # log-linear binning? Maybe just go back to external plotting...
    if(dimension=='1D'):
          plt.hist(self.cat_table[feature1][:], bins='auto')
          plt.xlabel(feature1)
          plt.ylabel('# of sources')
          plt.savefig(feature1+'_hist.png')
          plt.clf()
    elif(dimension=='2D'):
          plt.hist2d(self.cat_table[feature1][:], self.cat_table[feature2][:], bins=30)
          plt.xlabel(feature1)
          plt.ylabel(feature2)
          plt.savefig(feature1+'_'+feature2+'_hist.png')
          plt.clf()
    else:
         raise ValueError
    print('...Histogram saved.')



  def sort_feat(self,feature,sclass):
    #Sort the elements in the array of features according to a given source class 
    indexes=[]
    for i in range(self.Ns): 
       if(self.catdata.field('CLASS1')[i] in sclass):
          indexes.append(i)
    #indexes = np.where(self.catdata.field('CLASS1') in sclass) #does not work, xcheck
    sorted_feat=self.cat_table[feature][indexes]
    return sorted_feat
