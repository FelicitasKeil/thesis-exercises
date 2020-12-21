from astroquery.ned import Ned
import astropy.units as u
from astropy import coordinates
from astropy.table import Table
import pandas as pd
import numpy as np


def ned_query(ra, dec):
    # getting skycoordinate object
    co = coordinates.SkyCoord(ra=ra, dec=dec,
                              unit=(u.deg, u.deg), frame='icrs')

    search_radius = 0.01 * u.deg
    # this search radius is smaller than Fermi LAT resolution (please check)
    # get table with all objects inside the search radius
    result_table = Ned.query_region(co, radius=search_radius,
                                    equinox='J2000.0')
    result_table = Ned.query_object("NGC 224")

    print('names of the columns are ', result_table.colnames)
    print('to get an impression, here the whole table:')
    print(result_table)

    # get all the object names to get speific data
    object_names = result_table['Object Name']

    print('identified objects = ', object_names)

    # get table with positions of the first object as an example:
    position_table = Ned.get_table(object_names[0], table='positions')
    # position table is something you normally don't need but might be interesting
    # This should always work'
    # print('position table: ', position_table)

    # now the redshift is for many sources not available,will give you an error
    # therefore we will use try and exept
    for specific_object in object_names:
        try:
            redshift_table = Ned.get_table(specific_object, table='redshifts')
            print('redshift found four ', specific_object)
            print(redshift_table)
        except:
            print('no redshift for ', specific_object)
    
    return result_table


# choosing a random coordinate
ra_random_fermi_lat_source = 299.518
dec_random_fermi_lat_source = -38.784


table = ned_query(ra=ra_random_fermi_lat_source,
                  dec=dec_random_fermi_lat_source)
table_pd = pd.DataFrame(np.array(table))
