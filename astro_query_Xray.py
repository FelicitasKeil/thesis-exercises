from astroquery.ned import Ned
import astropy.units as u
from astropy import coordinates
from astroquery.esa.xmm_newton import XMMNewton


def xmmn_query(ra, dec):
    # getting skycoordinate object
    co = coordinates.SkyCoord(ra=ra, dec=dec,
                              unit=(u.deg, u.deg), frame='icrs')

    search_radius = 0.01 * u.deg
    # this search radius is smaller than the Fermi LAT resolution (check this)
    # get table with all objects inside the search radius, Julian date standard
    result_table = Ned.query_region(co, radius=search_radius,
                                    equinox='J2000.0')

    print('names of the columns are ', result_table.colnames)
    print('to get an impression, here the whole table:')
    print(result_table)

    # get all the object names to get speific data
    object_names = result_table['Object Name']

    print('identified objects = ', object_names)

    # get table with positions of the first object as an example:
    position_table = Ned.get_table(object_names[0], table='positions')
    # position table is sth. you normally don't need but might be interesting
    # This should always work
    print('position table: ')
    print(position_table)

    # now the redshift is for many sources not available. Thus an error...
    # therefore we will use try and except
    for specific_object in object_names:
        try:
            redshift_table = Ned.get_table(specific_object, table='redshifts')
            print('redshift found four ', specific_object)
            print(redshift_table)
        except:
            print('no redshift for ', specific_object)


# choosing a random coordinate
ra_random_fermi_lat_source = 299.518
dec_random_fermi_lat_source = -38.784


table = xmmn_query(ra=ra_random_fermi_lat_source, dec=dec_random_fermi_lat_source)
# XMMNewton.get_postcard('0505720401')
