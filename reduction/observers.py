from astropy.coordinates import EarthLocation

bernd = EarthLocation.from_geodetic(lon=13.708425, lat=51.003557, height=270)
bernd.info.name = "Dresden, Germany"

christian = EarthLocation.from_geodetic(lon=13.7, lat=51, height=300)
christian.info.name = "Dresden Gönnsdorf, Germany"

ulrich = EarthLocation.from_geodetic(lon=13.5, lat=52.5, height=100)
ulrich.info.name = "Berlin Marzahn, Germany"

filipe = EarthLocation.from_geodetic(lon=8.365, lat=+37.132)
filipe.info.name = "Armação de Pêra, Portugal"
