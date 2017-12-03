
from astropy.time import Time
from astropy.coordinates import SkyCoord

import astropy.units as u


from reduction.stars.variable_stars import RegularVariableObject


# delcep_Josi_orig = VariableStar("Josi initial period", Time(2457911, format='jd'), 5.366 * u.day)
delcep_Josi = RegularVariableObject("Josi", Time(2457911.0875, format='jd'), 5.366341 * u.day)
delcep_GCVS = RegularVariableObject("GCVS", Time(2436075.445, format='jd'), 5.366341 * u.day)
delcep_AAVSO = RegularVariableObject("AAVSO", Time(2436075.445, format='jd'), 5.366266 * u.day)

# wikipedia
delcep_pos= SkyCoord('22h29m10.26502s +58d24m54.7139s')
